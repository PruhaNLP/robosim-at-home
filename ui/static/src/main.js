import { createApp, nextTick, onMounted, reactive, watch } from "vue";
import { CAMERA_GAP, CAMERA_PAD, pickCameraLayout } from "./camera-layout.js";
import { SCENE_KEYS, SETTINGS_GROUPS, SETTINGS_NAV } from "./settings-schema.js";

// ======Settings=========
const STATS_POLL_MS = 250;
const COLLECT_POLL_MS = 500;
const CALIBRATE_POLL_MS = 100;
const RUNTIME_POLL_MS = 500;
const DEFAULT_TRAIN_BATCH = 8;
const CUSTOM_TRAIN_POLICY = "__custom__";
const SMOLVLA_MAX_CAMERAS = 5;
const API_TIMEOUT_MS = 45000;
const API_LONG_MS = 180000;
// ======Settings=========

function getPath(object, path) {
  return path.reduce((value, key) => value?.[key], object);
}

function setPath(object, path, value) {
  let cursor = object;
  for (const key of path.slice(0, -1)) cursor = cursor[key];
  cursor[path.at(-1)] = value;
}

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

function pageFromHash() {
  const hash = location.hash.replace("#", "");
  if (hash === "settings") return "settings";
  if (hash === "datasets") return "datasets";
  if (hash === "train") return "train";
  if (hash === "grpo") return "grpo";
  if (hash === "eval") return "eval";
  if (hash === "runs") return "runs";
  if (hash === "inference" || hash === "scene") return "inference";
  return "collect";
}

createApp({
  setup() {
    const state = reactive({
      page: pageFromHash(),
      section: "scene",
      openGroup: {},
      status: "Loading…",
      config: null,
      devices: [],
      modelDevices: [],
      viewToken: 0,
      stats: { frames: 0, hz: 0 },
      logs: [],
      dataset: {
        items: [],
        repoId: "local/so100_collect",
        newId: "",
        hubId: "",
        hub: { running: false, error: null, repoId: null, progress: null, logs: [] },
        hubPoll: null,
        episodes: [],
        episode: null,
      },
      collect: {
        step: "idle",
        message: "",
        help: [
          "Plug in only the lead arm USB and give it power.",
          "Do not unplug it yet.",
          "Click Start detect.",
        ],
        ports: [],
        motors: [],
        candidate: null,
        savedCalibration: null,
        busy: false,
        building: false,
        overlay: "",
        overlayError: false,
        scene: null,
        saved: null,
        logs: [],
        record: {
          running: false,
          recording: false,
          stopping: false,
          paused: false,
          episode: 0,
          frames: 0,
          fps: 15,
          task: "",
          repoId: "local/so100_collect",
          error: null,
        },
      },
      infer: {
        scene: null,
        saved: null,
        building: false,
        overlay: "Create a scene to run inference.",
        overlayError: false,
      },
      test: {
        allCheckpoints: [],
        duration: 20,
        nActionSteps: 50,
        numSteps: 10,
        controlHz: 15,
        viewFps: 15,
        checkpoint: "lerobot/smolvla_base",
        checkpoints: [{ id: "lerobot/smolvla_base", label: "lerobot/smolvla_base" }],
        running: false,
        phase: "idle",
        seconds: 0,
        success: null,
        error: null,
      },
      train: {
        policy: "lerobot/smolvla_base",
        custom: false,
        customPolicy: "",
        epochs: 20,
        batch: DEFAULT_TRAIN_BATCH,
        lr: 0.0001,
        saveEvery: 1,
        run: "smolvla",
        tune: "experts",
        repoIds: [],
        running: false,
        error: null,
        logs: [],
        progress: null,
        history: { loss: [], lr: [], gradNorm: [] },
      },
      grpo: {
        checkpoint: "lerobot/smolvla_base",
        run: "grpo",
        trainScope: "experts",
        groupSize: 16,
        parallel: 16,
        duration: 40,
        scenesPerUpdate: 16,
        sceneWaves: 4,
        sdeMode: "one_random_step",
        noiseLevel: 0.35,
        denoisingSteps: 10,
        expertLr: 5e-6,
        running: false,
        error: null,
        logs: [],
        progress: null,
        history: { reward: [], success: [], loss: [], clip: [], drift: [], kl: [], gradNorm: [] },
        scene: null,
        group: [],
        metrics: null,
      },
      eval: {
        checkpoint: "lerobot/smolvla_base",
        size: 32,
        seed: 10000,
        rerollSeed: 10000,
        duration: 40,
        parallel: 8,
        numSteps: 10,
        nActionSteps: 50,
        running: false,
        building: false,
        error: null,
        logs: [],
        progress: null,
        dataset: { size: 0, seed: 0, scenes: [] },
        results: [],
        metrics: null,
        selected: null,
      },
      runs: {
        items: [],
        selected: "",
        detail: null,
      },
      objects: { items: [] },
      catalogQuery: "",
    });

    let configRevision = 0;
    let statePoll = null;
    let stateInFlight = false;
    let hubWasRunning = false;
    let trainWasRunning = false;
    let grpoWasRunning = false;
    let evalWasRunning = false;
    let evalWasBuilding = false;
    let testWasRunning = false;
    const slotGens = { collect: null, infer: null, grpo: null };
    const SLOT_PAGE = { collect: "collect", infer: "inference", grpo: "grpo" };
    const SAVED_UNLOADED = "Saved scene is not loaded. Reset to load it.";
    const PAUSED_INFER = "Paused while inference is running.";

    function configPath(path) {
      return Array.isArray(path) ? path : [path];
    }

    const pendingPatches = new Map();
    const flushPersist = debounce(async (revision) => {
      const batch = [...pendingPatches.values()];
      pendingPatches.clear();
      if (!batch.length) return;
      state.status = "Saving…";
      try {
        let lastPayload = null;
        const slots = new Set();
        for (const item of batch) {
          const response = await api("/api/config", {
            method: "PUT",
            body: JSON.stringify({ path: item.path, value: fieldValue(item.path) }),
          });
          lastPayload = response;
          const skipRegen =
            item.path[0] === "compute"
            && (item.path[1] === "model"
              || item.path[1] === "policy_cameras"
              || item.path[1] === "policy_map");
          if (item.slot && SCENE_KEYS.has(item.path[0]) && !skipRegen) {
            slots.add(item.slot);
          }
        }
        if (lastPayload?.state) applySnapshot(lastPayload);
        else if (lastPayload?.config && revision === configRevision) state.config = lastPayload.config;
        applyGrpoDefaults(state.config);
        applyEvalDefaults(state.config);
        applyInferDefaults(state.config);
        state.status = "Saved";
        for (const slot of slots) generateScene(slot);
      } catch (error) {
        state.status = error.message;
      }
    }, 350);

    function persist(path, slot, revision) {
      const keys = configPath(path);
      pendingPatches.set(keys.join("."), { path: keys, slot });
      flushPersist(revision);
    }

    function overlayCollect(message, isError = false) {
      state.collect.overlay = message;
      state.collect.overlayError = Boolean(isError);
    }

    function overlayInfer(message, isError = false) {
      state.infer.overlay = message;
      state.infer.overlayError = Boolean(isError);
    }

    function groupsFor(section) {
      return SETTINGS_GROUPS[section] || [];
    }

    function isOpen(section, groupId) {
      if (state.openGroup[section] === undefined) return groupId === groupsFor(section)[0]?.id;
      return state.openGroup[section] === groupId;
    }

    function toggleGroup(section, groupId) {
      state.openGroup[section] = isOpen(section, groupId) ? "" : groupId;
    }

    function fieldValue(path) {
      return getPath(state.config, path);
    }

    function commit(path, value, slot) {
      if (JSON.stringify(fieldValue(path)) === JSON.stringify(value)) return;
      setPath(state.config, path, value);
      configRevision += 1;
      persist(path, slot, configRevision);
    }

    function commitNumber(path, rawValue, slot) {
      if (rawValue === "") return;
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      commit(path, value, slot);
    }

    function scrollLog(elementId) {
      nextTick(() => {
        const list = document.getElementById(elementId);
        if (list) list.scrollTop = list.scrollHeight;
      });
    }

    function addCollectLog(text, kind = "info") {
      state.collect.logs.push({ kind, text });
    }

    function addLog(text, kind = "info") {
      state.logs.push({ kind, text });
    }

    function isNetworkError(error) {
      const msg = String(error?.message || error || "");
      return (
        msg === "Failed to fetch"
        || msg === "Request timed out"
        || msg === "Load failed"
        || msg.startsWith("NetworkError")
      );
    }

    function bindLogScroll(source, elementId) {
      watch(
        () => [source().length, source().at(-1)?.text, source().at(-1)?.kind],
        () => scrollLog(elementId),
      );
    }

    bindLogScroll(() => state.collect.logs, "collect-log-list");
    bindLogScroll(() => state.train.logs, "train-log-list");
    bindLogScroll(() => state.grpo.logs, "grpo-log-list");
    bindLogScroll(() => state.eval.logs, "eval-log-list");
    bindLogScroll(() => state.logs, "log-list");
    watch(
      () => state.page,
      (page) => {
        if (page === "collect") scrollLog("collect-log-list");
        if (page === "train") scrollLog("train-log-list");
        if (page === "grpo") scrollLog("grpo-log-list");
        if (page === "eval") scrollLog("eval-log-list");
        if (page === "inference") scrollLog("log-list");
      },
    );

    function updateIndex(path, index, rawValue) {
      if (rawValue === "") return;
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      const current = fieldValue(path);
      if (current[index] === value) return;
      const next = current.slice();
      next[index] = value;
      commit(path, next);
    }

    function isAct() {
      return (state.config?.policy_mode || "smolvla") === "act";
    }

    function trainPolicies() {
      const mode = isAct() ? "act" : "smolvla";
      return (state.test.allCheckpoints || []).filter((item) => (item.type || "smolvla") === mode);
    }

    function trainPolicyValue() {
      if (state.train.custom) return String(state.train.customPolicy || "").trim();
      return String(state.train.policy || "").trim();
    }

    function trainPolicySelect() {
      if (state.train.custom) return CUSTOM_TRAIN_POLICY;
      const policy = String(state.train.policy || "").trim();
      if (isAct() && (policy === "act" || policy === "lerobot/act" || !policy)) return "act";
      if (trainPolicies().some((item) => item.id === policy)) return policy;
      if (policy) return CUSTOM_TRAIN_POLICY;
      return isAct() ? "act" : (trainPolicies()[0]?.id || CUSTOM_TRAIN_POLICY);
    }

    function setTrainPolicySelect(value) {
      if (value === CUSTOM_TRAIN_POLICY) {
        state.train.custom = true;
        const current = String(state.train.policy || "").trim();
        if (
          !String(state.train.customPolicy || "").trim()
          && current
          && current !== "act"
          && current !== "lerobot/act"
        ) {
          state.train.customPolicy = current;
        }
        return;
      }
      state.train.custom = false;
      state.train.policy = value;
    }

    function isCustomTrainPolicy() {
      return state.train.custom || trainPolicySelect() === CUSTOM_TRAIN_POLICY;
    }

    function trainPolicyReady() {
      return Boolean(trainPolicyValue());
    }

    function resolveActStart() {
      let policy = trainPolicyValue();
      let run = String(state.train.run || "").trim();
      if (!policy) policy = "act";
      if (!run || run === "smolvla") run = "act";
      return { policy, run };
    }

    function syncActTrainForm() {
      if (!isAct()) return;
      if (state.train.custom) {
        if (!state.train.run || state.train.run === "smolvla") state.train.run = "act";
        return;
      }
      const next = resolveActStart();
      const item = (state.test.allCheckpoints || []).find((entry) => entry.id === next.policy);
      if (next.policy === "lerobot/smolvla_base" || (item && item.type !== "act")) {
        next.policy = "act";
      }
      state.train.policy = next.policy;
      state.train.run = next.run;
    }

    function setPolicyMode(mode) {
      const next = mode === "act" ? "act" : "smolvla";
      const policy = String(state.train.policy || "").trim();
      const item = (state.test.allCheckpoints || []).find((entry) => entry.id === policy);
      if (next === "act") {
        if (policy === "lerobot/smolvla_base" || (item && item.type !== "act")) {
          state.train.policy = "act";
          state.train.custom = false;
        }
        if (state.train.run === "smolvla") state.train.run = "act";
        if (state.page === "grpo" || state.page === "eval") location.hash = "train";
        if (state.section === "grpo" || state.section === "eval") state.section = "training";
      } else {
        if (policy === "act" || policy === "lerobot/act" || (item && item.type === "act")) {
          state.train.policy = "lerobot/smolvla_base";
          state.train.custom = false;
        }
        if (state.train.run === "act") state.train.run = "smolvla";
      }
      commit(["policy_mode"], next);
      applyCheckpoints(state.test.allCheckpoints.length ? state.test.allCheckpoints : state.test.checkpoints);
      syncActTrainForm();
    }

    function settingsNav() {
      return SETTINGS_NAV.filter((item) => !isAct() || (item.id !== "grpo" && item.id !== "eval"));
    }

    function vec3List(path) {
      const value = fieldValue(path);
      return Array.isArray(value) ? value : [];
    }

    function addVec3(path) {
      commit(path, [...vec3List(path), [0, 0, 0]]);
    }

    function removeVec3(path, index) {
      commit(path, vec3List(path).filter((_, itemIndex) => itemIndex !== index));
    }

    function updateVec3List(path, row, col, rawValue) {
      if (rawValue === "") return;
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      const next = vec3List(path).map((item, itemIndex) => {
        const rowVals = Array.isArray(item) ? item.slice(0, 3) : [0, 0, 0];
        while (rowVals.length < 3) rowVals.push(0);
        if (itemIndex === row) rowVals[col] = value;
        return rowVals;
      });
      commit(path, next);
    }

    function catalogItems() {
      if (Array.isArray(state.objects.items)) return state.objects.items;
      const seen = new Map();
      for (const role of ["targets", "distractors"]) {
        for (const item of state.objects[role] || []) seen.set(item.id, item);
      }
      return [...seen.values()].sort((a, b) => a.id.localeCompare(b.id));
    }

    function catalogPreview(id) {
      return `/api/objects/preview?id=${encodeURIComponent(id)}`;
    }

    function catalogQueryText() {
      return String(state.catalogQuery || "").trim().toLowerCase();
    }

    function objectLabel(field, id) {
      const labels = fieldValue(field.labelsPath) || {};
      const aliases = labels[id];
      if (Array.isArray(aliases) && aliases[0]) return aliases[0];
      return id.replaceAll("_", " ");
    }

    function filteredCatalog(field) {
      const query = catalogQueryText();
      return catalogItems().filter((item) => {
        if (!query) return true;
        return (
          item.id.toLowerCase().includes(query)
          || objectLabel(field, item.id).toLowerCase().includes(query)
        );
      });
    }

    function catalogNode(role) {
      return fieldValue(["scene", role]) || {};
    }

    function catalogAll(role) {
      const node = catalogNode(role);
      if (node.all === undefined) return true;
      return Boolean(node.all);
    }

    function catalogInclude(role) {
      return catalogNode(role).include || [];
    }

    function isCatalogChecked(role, id) {
      return catalogAll(role) || catalogInclude(role).includes(id);
    }

    function catalogSelectedCount(role) {
      if (catalogAll(role)) return catalogItems().length;
      const include = new Set(catalogInclude(role));
      return catalogItems().filter((item) => include.has(item.id)).length;
    }

    function writeCatalog(role, all, include) {
      const node = { ...catalogNode(role) };
      node.all = all;
      node.include = include;
      commit(["scene", role], node);
    }

    function setCatalogChecked(role, id, checked) {
      const ids = catalogItems().map((item) => item.id);
      let include = catalogAll(role) ? ids : catalogInclude(role).filter((name) => ids.includes(name));
      if (checked && !include.includes(id)) include = [...include, id];
      if (!checked) include = include.filter((name) => name !== id);
      if (include.length === ids.length) writeCatalog(role, true, []);
      else writeCatalog(role, false, include);
    }

    function selectCatalogRole(role, checked) {
      if (!checked) return;
      const query = catalogQueryText();
      const visible = filteredCatalog({ labelsPath: ["language", "target_labels"] }).map((item) => item.id);
      if (!query) {
        writeCatalog(role, true, []);
        return;
      }
      writeCatalog(role, false, visible);
    }

    function setObjectLabel(field, id, value) {
      const labels = { ...(fieldValue(field.labelsPath) || {}) };
      const trimmed = String(value || "").trim();
      if (!trimmed) delete labels[id];
      else labels[id] = [trimmed];
      commit(field.labelsPath, labels);
    }

    function collectCameraNames() {
      return Array.isArray(state.collect.scene?.cameras) ? state.collect.scene.cameras : [];
    }

    function inferCameraNames() {
      return Array.isArray(state.infer.scene?.cameras) ? state.infer.scene.cameras : [];
    }

    function selectedCheckpoint() {
      return (state.test.checkpoints || []).find((item) => item.id === state.test.checkpoint) || null;
    }

    function policyMapSlots() {
      return ["camera1", "camera2", "camera3", "camera4", "camera5"];
    }

    function emptyPolicyMap() {
      return { camera1: "", camera2: "", camera3: "", camera4: "", camera5: "" };
    }

    function policyMapValue() {
      const raw = fieldValue(["compute", "policy_map"]);
      const mapping = emptyPolicyMap();
      if (raw && typeof raw === "object" && !Array.isArray(raw)) {
        for (const slot of policyMapSlots()) {
          mapping[slot] = String(raw[slot] || "").trim();
        }
        if (policyMapSlots().some((slot) => mapping[slot])) return mapping;
      }
      const list = fieldValue(["compute", "policy_cameras"]);
      if (Array.isArray(list)) {
        list.filter(Boolean).forEach((name, index) => {
          if (index < 5) mapping[policyMapSlots()[index]] = String(name);
        });
      }
      if (!policyMapSlots().some((slot) => mapping[slot])) {
        mapping.camera1 = "front";
        mapping.camera2 = "wrist";
      }
      return mapping;
    }

    function policyMapSlot(slot) {
      return policyMapValue()[slot] || "";
    }

    function policyMapChoices(slot) {
      const mounts = Object.keys(fieldValue(["cameras", "mounts"]) || {});
      const required = fieldValue(["cameras", "required"]) || [];
      const extra = ["front", "wrist", "overview", "top", "left", "right"];
      const used = new Set(
        policyMapSlots()
          .filter((name) => name !== slot)
          .map((name) => policyMapSlot(name))
          .filter(Boolean)
      );
      const names = [];
      const seen = new Set();
      for (const name of [...required, ...mounts, ...extra]) {
        if (!name || seen.has(name) || used.has(name)) continue;
        seen.add(name);
        names.push(name);
      }
      const current = policyMapSlot(slot);
      if (current && !names.includes(current)) names.unshift(current);
      return names;
    }

    function setPolicyMapSlot(slot, value) {
      const next = policyMapValue();
      const name = String(value || "").trim();
      if (name) {
        for (const other of policyMapSlots()) {
          if (other !== slot && next[other] === name) next[other] = "";
        }
      }
      next[slot] = name;
      if (!policyMapSlots().some((item) => next[item])) return;
      commit(["compute", "policy_map"], next);
    }

    function grpoCameraNames() {
      return Array.isArray(state.grpo.scene?.cameras) ? state.grpo.scene.cameras : [];
    }

    function pageSlot(page) {
      if (page === "collect") return "collect";
      if (page === "grpo") return "grpo";
      return "infer";
    }

    function collectTitle() {
      if (state.collect.step === "detect_unplug") return "Unplug the lead USB";
      if (state.collect.step === "detect_replug") return "Plug the lead USB back in";
      if (state.collect.step === "detected") return "Connect the lead";
      if (state.collect.step === "connected") return "Calibrate";
      if (state.collect.step === "calibrate_home") return "Set home";
      if (state.collect.step === "calibrate_range") return "Sweep the joints";
      return "Detect the lead";
    }

    function wizardLabel() {
      if (state.collect.step === "detect_unplug") return "USB unplugged";
      if (state.collect.step === "detect_replug") return "USB plugged in";
      if (state.collect.step === "detected") return "Connect";
      if (state.collect.step === "connected") return "Calibrate";
      if (state.collect.step === "calibrate_home") return "Home is set";
      if (state.collect.step === "calibrate_range") return "Finish calibration";
      return "Start detect";
    }

    function wizardNext() {
      if (state.collect.step === "detect_unplug") return detectUnplug();
      if (state.collect.step === "detect_replug") return detectReplug();
      if (state.collect.step === "detected") return connectLead();
      if (state.collect.step === "connected") return startCalibrate();
      if (state.collect.step === "calibrate_home") return setHome();
      if (state.collect.step === "calibrate_range") return finishCalibrate();
      return detectLead();
    }

    function applyLayout(page) {
      const stageId = page === "collect" ? "collect-stage" : page === "grpo" ? "grpo-stage" : "infer-stage";
      const gridId = page === "collect" ? "collect-cameras" : page === "grpo" ? "grpo-cameras" : "infer-cameras";
      const stage = document.getElementById(stageId);
      const grid = document.getElementById(gridId);
      if (!stage || !grid || state.page !== page) return;
      const names = page === "collect"
        ? collectCameraNames()
        : page === "grpo"
          ? grpoCameraNames()
          : inferCameraNames();
      const tiles = [...grid.querySelectorAll(".camera")];
      if (!names.length || tiles.length !== names.length) return;
      const layout = pickCameraLayout(
        names.length,
        Math.max(0, stage.clientWidth - CAMERA_PAD * 2),
        Math.max(0, stage.clientHeight - CAMERA_PAD * 2)
      );
      if (!layout) return;
      grid.style.width = `${layout.cols * layout.tile + CAMERA_GAP * (layout.cols - 1)}px`;
      grid.style.height = `${layout.rows * layout.tile + CAMERA_GAP * (layout.rows - 1)}px`;
      tiles.forEach((tile, index) => {
        const [column, row] = layout.positions[index];
        tile.style.width = `${layout.tile}px`;
        tile.style.height = `${layout.tile}px`;
        tile.style.left = `${column * (layout.tile + CAMERA_GAP)}px`;
        tile.style.top = `${row * (layout.tile + CAMERA_GAP)}px`;
      });
    }

    function scheduleLayout(page) {
      requestAnimationFrame(() => {
        applyLayout(page);
        requestAnimationFrame(() => applyLayout(page));
      });
    }

    function cameraImgs(page) {
      const id = page === "collect" ? "collect-cameras" : page === "grpo" ? "grpo-cameras" : "infer-cameras";
      return [...document.querySelectorAll(`#${id} .camera img`)];
    }

    function streamSrc(name, page) {
      const slot = pageSlot(page);
      const q = `camera=${encodeURIComponent(name)}&slot=${slot}&s=${state.viewToken}`;
      return `/api/view?${q}`;
    }

    function clearCameraSrcs(page) {
      cameraImgs(page).forEach((img) => {
        img.dataset.stream = "";
        img.src = "";
        if (img._blobUrl) {
          URL.revokeObjectURL(img._blobUrl);
          img._blobUrl = "";
        }
      });
    }

    function paintInferViews(frameGen) {
      const names = inferCameraNames();
      if (!names.length) return;
      cameraImgs("inference").forEach((img, index) => {
        const name = names[index];
        if (!name) return;
        const next = `/api/view?camera=${encodeURIComponent(name)}&slot=infer&g=${frameGen}`;
        if (img.dataset.stream === next) return;
        img.dataset.stream = next;
        img.src = next;
      });
    }

    function applyCameraSrcs(page) {
      if (page === "inference" && state.test.running) {
        paintInferViews(Date.now());
        return;
      }
      const names = page === "collect"
        ? collectCameraNames()
        : page === "grpo"
          ? grpoCameraNames()
          : inferCameraNames();
      cameraImgs(page).forEach((img, index) => {
        const name = names[index];
        if (!name) return;
        const next = streamSrc(name, page);
        if (img.dataset.stream === next) return;
        img.dataset.stream = next;
        img.src = next;
      });
    }

    let collectViewTimer = null;
    let collectViewBusy = false;

    function stopCollectViews() {
      if (collectViewTimer) {
        clearInterval(collectViewTimer);
        collectViewTimer = null;
      }
      collectViewBusy = false;
      clearCameraSrcs("collect");
    }

    async function tickCollectViews() {
      if (collectViewBusy || state.page !== "collect" || state.collect.step !== "ready") {
        return;
      }
      collectViewBusy = true;
      try {
        const names = collectCameraNames();
        const imgs = cameraImgs("collect");
        await Promise.all(
          names.map(async (name, index) => {
            const img = imgs[index];
            if (!img) return;
            const response = await fetch(
              `/api/view?camera=${encodeURIComponent(name)}&slot=collect&t=${Date.now()}`
            );
            if (!response.ok) return;
            const blob = await response.blob();
            if (!blob.size) return;
            if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = URL.createObjectURL(blob);
            img.src = img._blobUrl;
          })
        );
      } catch {
        return;
      } finally {
        collectViewBusy = false;
      }
    }

    function startCollectViews() {
      stopCollectViews();
      const fps = Math.max(1, Math.min(30, Number(state.collect.record.fps) || 15));
      tickCollectViews();
      collectViewTimer = setInterval(tickCollectViews, Math.round(1000 / fps));
    }

    let inferViewTimer = null;
    let inferViewBusy = false;

    function stopInferViews() {
      if (inferViewTimer) {
        clearInterval(inferViewTimer);
        inferViewTimer = null;
      }
      inferViewBusy = false;
    }

    async function tickInferViews() {
      if (inferViewBusy || state.page !== "inference" || !state.test.running) {
        return;
      }
      inferViewBusy = true;
      try {
        const names = inferCameraNames();
        const imgs = cameraImgs("inference");
        await Promise.all(
          names.map(async (name, index) => {
            const img = imgs[index];
            if (!img) return;
            const response = await fetch(
              `/api/view?camera=${encodeURIComponent(name)}&slot=infer&t=${Date.now()}`
            );
            if (!response.ok) return;
            const blob = await response.blob();
            if (!blob.size) return;
            if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = URL.createObjectURL(blob);
            img.src = img._blobUrl;
          })
        );
      } catch {
        return;
      } finally {
        inferViewBusy = false;
      }
    }

    function startInferViews() {
      if (inferViewTimer) return;
      const fps = Math.max(1, Math.min(30, Number(state.test.viewFps) || 15));
      tickInferViews();
      inferViewTimer = setInterval(tickInferViews, Math.round(1000 / fps));
    }

    let grpoViewTimer = null;
    let grpoViewBusy = false;

    function stopGrpoViews() {
      if (grpoViewTimer) {
        clearInterval(grpoViewTimer);
        grpoViewTimer = null;
      }
      grpoViewBusy = false;
    }

    async function tickGrpoViews() {
      if (grpoViewBusy || state.page !== "grpo" || !state.grpo.running) {
        return;
      }
      grpoViewBusy = true;
      try {
        const names = grpoCameraNames();
        const imgs = cameraImgs("grpo");
        await Promise.all(
          names.map(async (name, index) => {
            const img = imgs[index];
            if (!img) return;
            const response = await fetch(
              `/api/view?camera=${encodeURIComponent(name)}&slot=grpo&t=${Date.now()}`
            );
            if (!response.ok) return;
            const blob = await response.blob();
            if (!blob.size) return;
            if (img._blobUrl) URL.revokeObjectURL(img._blobUrl);
            img._blobUrl = URL.createObjectURL(blob);
            img.src = img._blobUrl;
          })
        );
      } catch {
        return;
      } finally {
        grpoViewBusy = false;
      }
    }

    function startGrpoViews() {
      if (grpoViewTimer) return;
      const fps = Math.max(
        1,
        Math.min(30, Number(state.config?.environment?.grpo?.visualization?.stream_fps) || 15)
      );
      tickGrpoViews();
      grpoViewTimer = setInterval(tickGrpoViews, Math.round(1000 / fps));
    }

    function preloadViews(names, page) {
      const slot = pageSlot(page);
      return Promise.all(
        names.map(
          (name) =>
            new Promise((resolve) => {
              const img = new Image();
              img.onload = () => resolve();
              img.onerror = () => resolve();
              img.src = `/api/view?camera=${encodeURIComponent(name)}&slot=${slot}&s=${state.viewToken}`;
            })
        )
      );
    }

    async function refreshViews(page) {
      state.viewToken += 1;
      const names = page === "collect"
        ? collectCameraNames()
        : page === "grpo"
          ? grpoCameraNames()
          : inferCameraNames();
      if (!names.length) {
        if (page === "collect") overlayCollect("No cameras in this scene", true);
        else if (page !== "grpo") overlayInfer("No cameras in this scene", true);
        return;
      }
      await nextTick();
      applyLayout(page);
      if (page === "collect") {
        startCollectViews();
      } else if (page === "inference" && state.test.running) {
        startInferViews();
      } else if (page === "grpo" && state.grpo.running) {
        startGrpoViews();
      } else {
        stopInferViews();
        stopGrpoViews();
        await preloadViews(names, page);
        applyCameraSrcs(page);
      }
      syncSlotOverlays();
      scheduleLayout(page);
    }

    function resolveSnapshot(payload) {
      if (!payload) return null;
      if (payload.state) return payload.state;
      if (payload.slots || payload.worker) return payload;
      return null;
    }

    function logsChanged(prev, next) {
      if (!Array.isArray(next)) return false;
      return (
        prev.length !== next.length
        || (next.length > 0
          && (prev[prev.length - 1]?.text !== next[next.length - 1]?.text
            || prev[prev.length - 1]?.kind !== next[next.length - 1]?.kind))
      );
    }

    function liveSlotScene(slot) {
      if (slot === "collect") return Boolean(state.collect.scene);
      if (slot === "grpo") return Boolean(state.grpo.scene);
      return Boolean(state.infer.scene);
    }

    function syncSlotOverlays() {
      if (state.collect.record.paused) {
        overlayCollect(PAUSED_INFER);
      } else if (
        !state.collect.scene
        && state.collect.saved
        && state.page === "collect"
        && !state.collect.building
        && !state.collect.overlayError
      ) {
        overlayCollect(SAVED_UNLOADED);
      } else if (
        state.collect.scene
        && !state.collect.building
        && !state.collect.overlayError
        && (state.collect.overlay === SAVED_UNLOADED || state.collect.overlay === PAUSED_INFER)
      ) {
        overlayCollect("");
      }

      if (
        !state.infer.scene
        && state.infer.saved
        && state.page === "inference"
        && !state.infer.building
        && !state.test.running
        && !state.infer.overlayError
      ) {
        overlayInfer(SAVED_UNLOADED);
      } else if (
        state.infer.scene
        && !state.infer.building
        && !state.test.running
        && !state.infer.overlayError
        && state.infer.overlay === SAVED_UNLOADED
      ) {
        overlayInfer("");
      }
    }

    function statusText() {
      if (state.collect.record.recording) return "Recording…";
      if (state.collect.record.stopping) return "Saving…";
      if (state.collect.building || state.infer.building) return "Building…";
      if (state.collect.busy && state.collect.step !== "ready") return "Working…";
      if (state.test.running) {
        if (state.test.phase === "loading") return isAct() ? "Loading ACT…" : "Loading SmolVLA…";
        if (state.test.phase === "running") return "Running test…";
        return "Starting test…";
      }
      if (state.train.running) return "Training…";
      if (state.grpo.running) return "GRPO…";
      if (state.eval.building) return "Building valset…";
      if (state.eval.running) return "Eval…";
      if (state.dataset.hub.running) return state.dataset.hub.progress?.label || "Importing…";
      return state.status || "Ready";
    }

    function applySnapshot(payload, options = {}) {
      const boot = Boolean(options.boot);
      const drafts = Boolean(options.drafts) || boot;
      const snap = resolveSnapshot(payload);
      const recOnly = !snap && drafts ? payload?.record : null;
      if (!snap && !recOnly) return;

      if (snap) {
        if (snap.config && (boot || pendingPatches.size === 0)) {
          state.config = snap.config;
        }
        if (Array.isArray(snap.devices)) state.devices = snap.devices;
        if (Array.isArray(snap.modelDevices)) state.modelDevices = snap.modelDevices;
        if (snap.objects) state.objects = snap.objects;

        const collect = snap.collect || {};
        if (collect.step != null) state.collect.step = collect.step;
        if (collect.message != null) state.collect.message = collect.message || "";
        if (Array.isArray(collect.help) && collect.help.length) state.collect.help = collect.help;
        if (Array.isArray(collect.ports)) state.collect.ports = collect.ports;
        if (Array.isArray(collect.motors)) state.collect.motors = collect.motors;
        if ("candidate" in collect) state.collect.candidate = collect.candidate || null;
        if ("savedCalibration" in collect) {
          state.collect.savedCalibration = collect.savedCalibration || null;
        }

        const slots = snap.slots || {};
        if (slots.collect) {
          state.collect.scene = slots.collect.loaded ? slots.collect.scene : null;
          state.collect.saved = slots.collect.saved || null;
        }
        if (slots.infer) {
          const nextScene = slots.infer.loaded ? slots.infer.scene : null;
          const prev = state.infer.scene;
          const testRunning = Boolean(snap.test && snap.test.running);
          if (nextScene) {
            if (
              !prev
              || prev.seed !== nextScene.seed
              || prev.instruction !== nextScene.instruction
              || (prev.cameras || []).join("\0") !== (nextScene.cameras || []).join("\0")
            ) {
              state.infer.scene = nextScene;
            }
          } else if (!testRunning && !state.infer.building) {
            state.infer.scene = null;
          }
          state.infer.saved = slots.infer.saved || null;
          if (testRunning && state.page === "inference" && slots.infer.frameGen != null) {
            paintInferViews(slots.infer.frameGen);
          }
        }
        if (slots.grpo) {
          state.grpo.scene = slots.grpo.loaded ? slots.grpo.scene : null;
        }

        const rec = collect.record;
        if (rec) {
          state.collect.record.running = rec.running;
          state.collect.record.recording = rec.recording;
          state.collect.record.stopping = Boolean(rec.stopping);
          state.collect.record.paused = Boolean(rec.paused);
          state.collect.record.episode = rec.episode;
          state.collect.record.frames = rec.frames;
          if ("error" in rec) state.collect.record.error = rec.error || null;
          if (logsChanged(state.collect.logs, rec.logs)) state.collect.logs = rec.logs;
          if (drafts) {
            if (rec.fps != null) state.collect.record.fps = rec.fps;
            if (rec.repoId != null) {
              state.collect.record.repoId = rec.repoId;
              state.dataset.repoId = rec.repoId;
            }
            if (rec.task != null) state.collect.record.task = rec.task;
          }
        }

        if (boot && !state.collect.record.task && state.collect.scene?.instruction) {
          state.collect.record.task = state.collect.scene.instruction;
        }

        const train = snap.train;
        if (train) {
          const running = Boolean(train.running);
          state.train.running = running;
          state.train.error = train.error || null;
          if (train.progress) state.train.progress = train.progress;
          if (train.history) state.train.history = train.history;
          if (logsChanged(state.train.logs, train.logs)) state.train.logs = train.logs;
          if (drafts) {
            if (train.policy) state.train.policy = train.policy;
            if (train.epochs) state.train.epochs = train.epochs;
            if (train.batch) state.train.batch = train.batch;
            if (train.lr) state.train.lr = train.lr;
            if (train.saveEvery != null) state.train.saveEvery = train.saveEvery;
            if (train.run) state.train.run = train.run;
            if (train.tune) state.train.tune = train.tune;
            if (Array.isArray(train.repoIds)) state.train.repoIds = train.repoIds;
          }
          if (trainWasRunning && !running) {
            loadCheckpoints();
            if (train.error) state.status = train.error;
            else if (state.page === "train") state.status = "Ready";
          }
          trainWasRunning = running;
        }

        if (boot) {
          applyGrpoDefaults(snap.config || state.config);
          applyEvalDefaults(snap.config || state.config);
          applyInferDefaults(snap.config || state.config);
        }

        const ev = snap.eval;
        if (ev) {
          const running = Boolean(ev.running);
          const building = Boolean(ev.building);
          state.eval.running = running;
          state.eval.building = building;
          state.eval.error = ev.error || null;
          if (ev.progress) state.eval.progress = ev.progress;
          if (ev.dataset) state.eval.dataset = ev.dataset;
          if (Array.isArray(ev.results)) state.eval.results = ev.results;
          state.eval.metrics = ev.metrics || null;
          if (logsChanged(state.eval.logs, ev.logs)) state.eval.logs = ev.logs;
          if (drafts) {
            if (ev.checkpoint) state.eval.checkpoint = ev.checkpoint;
            if (ev.size) state.eval.size = ev.size;
            if (ev.seed != null) state.eval.seed = ev.seed;
            if (ev.duration) state.eval.duration = ev.duration;
            if (ev.parallel) state.eval.parallel = ev.parallel;
            if (ev.numSteps) state.eval.numSteps = ev.numSteps;
            if (ev.nActionSteps) state.eval.nActionSteps = ev.nActionSteps;
          }
          const scenes = ev.dataset?.scenes || [];
          if (state.eval.selected == null && scenes.length) {
            state.eval.selected = scenes[0].index;
            state.eval.rerollSeed = scenes[0].seed;
          }
          if (state.eval.selected != null && !scenes.some((item) => item.index === state.eval.selected)) {
            state.eval.selected = scenes[0] ? scenes[0].index : null;
            state.eval.rerollSeed = scenes[0] ? scenes[0].seed : state.eval.seed;
          }
          if (evalWasRunning && !running) {
            if (ev.error) state.status = ev.error;
            else if (state.page === "eval") state.status = "Ready";
          }
          if (evalWasBuilding && !building) {
            if (ev.error) state.status = ev.error;
            else if (state.page === "eval") state.status = "Ready";
          }
          evalWasRunning = running;
          evalWasBuilding = building;
        }

        const grpo = snap.grpo;
        if (grpo) {
          const running = Boolean(grpo.running);
          state.grpo.running = running;
          state.grpo.error = grpo.error || null;
          if (grpo.progress) state.grpo.progress = grpo.progress;
          if (grpo.history) state.grpo.history = grpo.history;
          if (Array.isArray(grpo.group)) state.grpo.group = grpo.group;
          if (grpo.metrics) state.grpo.metrics = grpo.metrics;
          if (logsChanged(state.grpo.logs, grpo.logs)) state.grpo.logs = grpo.logs;
          if (drafts) {
            if (grpo.checkpoint) state.grpo.checkpoint = grpo.checkpoint;
            if (grpo.run) state.grpo.run = grpo.run;
          }
          if (grpoWasRunning && !running) {
            loadCheckpoints();
            stopGrpoViews();
            if (grpo.error) state.status = grpo.error;
            else if (state.page === "grpo") state.status = "Ready";
          }
          if (running && state.page === "grpo" && state.grpo.scene) startGrpoViews();
          grpoWasRunning = running;
        }

        const test = snap.test;
        if (test) {
          if (logsChanged(state.logs, test.logs)) state.logs = test.logs;
          const running = Boolean(test.running);
          state.test.running = running;
          if (test.phase != null) state.test.phase = test.phase;
          if (test.seconds != null) state.test.seconds = test.seconds;
          if ("success" in test) state.test.success = test.success;
          if ("error" in test) state.test.error = test.error;
          if (running) {
            testWasRunning = true;
            overlayInfer("");
            if (state.page === "inference") {
              startInferViews();
              if (slots.infer?.frameGen != null) paintInferViews(slots.infer.frameGen);
            }
          } else {
            const ended = testWasRunning;
            testWasRunning = false;
            stopInferViews();
            if (ended) {
              if (state.page !== "inference") {
                if (test.phase === "error") state.status = "Failed";
                else state.status = test.success ? "Success" : "Done";
              } else {
                clearCameraSrcs("inference");
                refreshViews("inference").then(() => {
                  if (test.phase === "error") {
                    overlayInfer(test.error || "Test failed", true);
                    state.status = "Failed";
                    return;
                  }
                  overlayInfer("");
                  state.status = test.success ? "Success" : "Done";
                });
              }
            }
          }
        }

        if (Array.isArray(snap.infer?.checkpoints)) applyCheckpoints(snap.infer.checkpoints);
        applyHub(snap);
        if (snap.stats) {
          if (state.stats.frames !== snap.stats.frames || state.stats.hz !== snap.stats.hz) {
            state.stats.frames = snap.stats.frames;
            state.stats.hz = snap.stats.hz;
          }
        }

        for (const slot of ["collect", "infer", "grpo"]) {
          const view = slots[slot];
          if (!view || view.gen == null) continue;
          const prev = slotGens[slot];
          slotGens[slot] = view.gen;
          if (!boot && prev !== view.gen && state.page === SLOT_PAGE[slot] && liveSlotScene(slot)) {
            refreshViews(SLOT_PAGE[slot]);
          }
        }

        if (state.page === "collect" && state.collect.step === "ready") {
          if (state.collect.scene && !collectViewTimer) startCollectViews();
          if (!state.collect.scene) stopCollectViews();
        }

        syncSlotOverlays();
        return;
      }

      state.collect.record.running = recOnly.running;
      state.collect.record.recording = recOnly.recording;
      state.collect.record.stopping = Boolean(recOnly.stopping);
      state.collect.record.paused = Boolean(recOnly.paused);
      state.collect.record.episode = recOnly.episode;
      state.collect.record.frames = recOnly.frames;
      if (recOnly.fps != null) state.collect.record.fps = recOnly.fps;
      if (recOnly.repoId != null) {
        state.collect.record.repoId = recOnly.repoId;
        state.dataset.repoId = recOnly.repoId;
      }
      if (recOnly.task != null) state.collect.record.task = recOnly.task;
    }

    function pollIntervalMs() {
      const step = String(state.collect.step || "");
      if (step.startsWith("calibrate")) return CALIBRATE_POLL_MS;
      if (step !== "ready") return COLLECT_POLL_MS;
      return RUNTIME_POLL_MS;
    }

    function pollStateUrl() {
      return String(state.collect.step || "") !== "ready" ? "/api/state?scan=1" : "/api/state";
    }

    function stopStatePoll() {
      if (statePoll) {
        clearTimeout(statePoll);
        statePoll = null;
      }
    }

    function scheduleStatePoll() {
      stopStatePoll();
      statePoll = setTimeout(async () => {
        await pollState();
        scheduleStatePoll();
      }, pollIntervalMs());
    }

    async function pollState() {
      if (stateInFlight) return;
      stateInFlight = true;
      try {
        applySnapshot(await api(pollStateUrl()));
      } catch {
        return;
      } finally {
        stateInFlight = false;
      }
    }

    async function generateScene(slot) {
      const isCollect = slot === "collect";
      const block = isCollect ? state.collect : state.infer;
      if (block.building || !state.config) return;
      if (isCollect && (state.collect.record.recording || state.collect.record.stopping)) return;
      if (!isCollect) {
        testWasRunning = false;
        state.test.running = false;
      }
      block.building = true;
      if (isCollect) overlayCollect("Building scene…");
      else overlayInfer("Building scene…");
      state.status = "Building…";
      if (isCollect) addCollectLog("Building scene…");
      else addLog("Building scene…");
      try {
        const payload = await api("/api/generate", {
          method: "POST",
          body: JSON.stringify({
            seed: state.config.seed,
            slot,
          }),
          timeout: API_LONG_MS,
        });
        applySnapshot(payload);
        if (isCollect) {
          overlayCollect("");
          if (state.collect.scene?.instruction) {
            state.collect.record.task = state.collect.scene.instruction;
            saveCollectSettings();
          }
          addCollectLog(`Scene ${state.collect.scene?.seed ?? payload.scene?.seed ?? ""} ready.`);
          if (state.collect.scene) await refreshViews("collect");
        } else {
          overlayInfer("");
          addLog(`Scene ${state.infer.scene?.seed ?? payload.scene?.seed ?? ""} ready.`);
          if (state.infer.scene) await refreshViews("inference");
        }
        state.status = "Ready";
      } catch (error) {
        if (isCollect) {
          overlayCollect(error.message, true);
          addCollectLog(error.message, "error");
        } else {
          overlayInfer(error.message, true);
          addLog(error.message, "error");
        }
        state.status = "Failed";
      } finally {
        block.building = false;
      }
    }

    async function regenerate(slot) {
      if (slot === "collect" && (state.collect.record.recording || state.collect.record.stopping)) return;
      const block = slot === "collect" ? state.collect : state.infer;
      if (block.building) return;
      state.config.seed = Math.floor(Math.random() * 1_000_000_000);
      configRevision += 1;
      state.status = "Saving…";
      try {
        const response = await api("/api/config", {
          method: "PUT",
          body: JSON.stringify({ path: ["seed"], value: state.config.seed }),
        });
        if (response.state) applySnapshot(response);
        else if (response.config) state.config = response.config;
        await generateScene(slot);
      } catch (error) {
        state.status = error.message;
      }
    }

    async function resetScene(slot) {
      const isCollect = slot === "collect";
      const block = isCollect ? state.collect : state.infer;
      if (block.building || !state.config) return;
      if (isCollect && (state.collect.record.recording || state.collect.record.stopping || (!state.collect.scene && !state.collect.saved))) return;
      if (!isCollect && !state.infer.scene && !state.infer.saved) return;
      if (!isCollect) {
        testWasRunning = false;
        state.test.running = false;
      }
      block.building = true;
      if (isCollect) overlayCollect("Resetting scene…");
      else overlayInfer("Resetting scene…");
      state.status = "Resetting…";
      try {
        const payload = await api("/api/reset", {
          method: "POST",
          body: JSON.stringify({ slot }),
          timeout: API_LONG_MS,
        });
        applySnapshot(payload);
        if (isCollect) {
          overlayCollect("");
          if (state.collect.scene?.instruction) {
            state.collect.record.task = state.collect.scene.instruction;
            saveCollectSettings();
          }
          addCollectLog("Scene reset.");
          if (state.collect.scene) await refreshViews("collect");
        } else {
          overlayInfer("");
          addLog("Scene reset.");
          if (state.infer.scene) await refreshViews("inference");
        }
        state.status = "Ready";
      } catch (error) {
        if (isCollect) {
          overlayCollect(error.message, true);
          addCollectLog(error.message, "error");
        } else {
          overlayInfer(error.message, true);
          addLog(error.message, "error");
        }
        state.status = error.message;
      } finally {
        block.building = false;
      }
    }

    async function leaderAction(path) {
      state.collect.busy = true;
      state.collect.message = "Working…";
      state.status = "Working…";
      try {
        const payload = await api(path, { method: "POST", body: "{}" });
        applySnapshot(payload);
        if (state.collect.scene) {
          if (state.collect.scene.instruction) {
            state.collect.record.task = state.collect.scene.instruction;
            saveCollectSettings();
          }
          addCollectLog("Scene ready.");
          await refreshViews("collect");
        }
        state.status = payload.message || "Ready";
      } catch (error) {
        state.collect.message = error.message;
        state.status = error.message;
        addCollectLog(error.message, "error");
      } finally {
        state.collect.busy = false;
      }
    }

    function detectLead() {
      return leaderAction("/api/leader/detect");
    }

    function detectUnplug() {
      return leaderAction("/api/leader/detect/unplug");
    }

    function detectReplug() {
      return leaderAction("/api/leader/detect/replug");
    }

    function connectLead() {
      return leaderAction("/api/leader/connect");
    }

    function useSavedCalibration() {
      return leaderAction("/api/leader/use_calibration");
    }

    function startCalibrate() {
      return leaderAction("/api/leader/calibrate/start");
    }

    function setHome() {
      return leaderAction("/api/leader/calibrate/home");
    }

    function finishCalibrate() {
      return leaderAction("/api/leader/calibrate/finish");
    }

    async function saveCollectSettings() {
      try {
        const payload = await api("/api/collect/settings", {
          method: "POST",
          body: JSON.stringify({
            fps: Number(state.collect.record.fps) || 15,
            repo_id: state.collect.record.repoId,
            task: state.collect.record.task,
          }),
        });
        applySnapshot(payload, { drafts: true });
      } catch (error) {
        state.status = error.message;
      }
    }

    async function startRecord() {
      if (
        !state.collect.scene
        || state.collect.record.recording
        || state.collect.record.stopping
      ) return;
      state.collect.busy = true;
      try {
        const payload = await api("/api/collect/record", {
          method: "POST",
          body: JSON.stringify({
            fps: Number(state.collect.record.fps) || 15,
            repo_id: state.collect.record.repoId,
            task: state.collect.record.task || state.collect.scene.instruction,
          }),
        });
        applySnapshot(payload);
        state.status = "Recording…";
        addCollectLog("Recording…");
      } catch (error) {
        addCollectLog(error.message, "error");
        state.status = error.message;
      } finally {
        state.collect.busy = false;
      }
    }

    async function stopRecord(save) {
      if (state.collect.record.stopping || !state.collect.record.recording) return;
      state.collect.record.stopping = true;
      state.collect.busy = true;
      state.status = save ? "Saving…" : "Discarding…";
      try {
        const payload = await api("/api/collect/record/stop", {
          method: "POST",
          body: JSON.stringify({ save }),
        });
        applySnapshot(payload);
        state.status = save ? "Saved" : "Discarded";
        if (state.page === "datasets") await loadEpisodes();
      } catch (error) {
        addCollectLog(error.message, "error");
        state.status = error.message;
        state.collect.record.stopping = false;
      } finally {
        state.collect.busy = false;
      }
    }

    function episodeVideo(item, camera) {
      const target = encodeURIComponent(item.id || item.index);
      const repo = encodeURIComponent(state.dataset.repoId);
      return `/api/episodes/${target}/video/${encodeURIComponent(camera)}?repo_id=${repo}`;
    }

    function datasetStats() {
      const episodes = state.dataset.episodes || [];
      const totalEpisodes = episodes.length;
      const totalFrames = episodes.reduce((acc, ep) => acc + (ep.frames || 0), 0);
      const totalSeconds = episodes.reduce((acc, ep) => acc + (ep.seconds || 0), 0);
      return {
        totalEpisodes,
        totalFrames,
        totalSeconds: Math.round(totalSeconds * 10) / 10,
      };
    }

    function applyHub(payload) {
      const hub = payload?.hub || payload?.hubImport;
      if (!hub) return;
      const running = Boolean(hub.running);
      state.dataset.hub.running = running;
      state.dataset.hub.error = hub.error || null;
      state.dataset.hub.repoId = hub.repoId || null;
      if (hub.progress) state.dataset.hub.progress = hub.progress;
      if (Array.isArray(hub.logs)) state.dataset.hub.logs = hub.logs;
      if (hubWasRunning && !running) {
        if (hub.error) state.status = hub.error;
        else {
          state.status = "Ready";
          loadEpisodes();
        }
      }
      hubWasRunning = running;
    }

    function applyDatasets(payload) {
      if (!payload) return;
      if (Array.isArray(payload.datasets)) state.dataset.items = payload.datasets;
      if (payload.repoId) {
        state.dataset.repoId = payload.repoId;
        state.collect.record.repoId = payload.repoId;
      }
      if (Array.isArray(payload.episodes)) {
        state.dataset.episodes = payload.episodes;
        if (
          state.dataset.episode
          && !payload.episodes.some((item) => (item.id && item.id === state.dataset.episode.id) || item.index === state.dataset.episode.index)
        ) {
          state.dataset.episode = null;
        }
      }
      applyHub(payload);
    }

    function isHubDataset(repoId) {
      const id = repoId || state.dataset.repoId;
      const item = (state.dataset.items || []).find((row) => row.repoId === id);
      return item?.source === "hub";
    }

    async function loadDatasets(options = {}) {
      try {
        const payload = await api("/api/datasets");
        applyDatasets(payload);
        if (options.skipEpisodes) return;
        await loadEpisodes();
      } catch (error) {
        state.status = error.message;
      }
    }

    async function loadEpisodes() {
      try {
        const repo = encodeURIComponent(state.dataset.repoId);
        const payload = await api(`/api/episodes?repo_id=${repo}`);
        applyDatasets(payload);
      } catch (error) {
        state.status = error.message;
      }
    }

    async function selectCollectDataset() {
      await selectDataset(state.collect.record.repoId);
    }

    async function selectDataset(repoId) {
      try {
        const payload = await api("/api/datasets/select", {
          method: "POST",
          body: JSON.stringify({ repo_id: repoId }),
        });
        applyDatasets(payload);
        applySnapshot(payload, { drafts: true });
        state.status = "Ready";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function createDataset() {
      const repoId = String(state.dataset.newId || "").trim();
      if (!repoId) return;
      try {
        const payload = await api("/api/datasets", {
          method: "POST",
          body: JSON.stringify({ repo_id: repoId }),
        });
        state.dataset.newId = "";
        applyDatasets(payload);
        applySnapshot(payload);
        await loadEpisodes();
        state.status = "Ready";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function importHubDataset() {
      const repoId = String(state.dataset.hubId || "").trim();
      if (!repoId.includes("/") || repoId.startsWith("/") || repoId.endsWith("/")) {
        state.status = "HF dataset must look like org/name";
        return;
      }
      try {
        const payload = await api("/api/datasets/hub", {
          method: "POST",
          body: JSON.stringify({ repo_id: repoId }),
        });
        state.dataset.hubId = "";
        applyDatasets(payload);
        applySnapshot(payload);
        state.status = payload.hub?.progress?.label || "Importing…";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function syncHubDataset() {
      if (!state.dataset.repoId || state.collect.record.recording || state.dataset.hub.running) return;
      try {
        const payload = await api("/api/datasets/sync", {
          method: "POST",
          body: JSON.stringify({ repo_id: state.dataset.repoId }),
        });
        applyDatasets(payload);
        applySnapshot(payload);
        state.dataset.episode = null;
        state.status = payload.hub?.progress?.label || "Syncing…";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function deleteDataset() {
      if (!state.dataset.repoId || state.collect.record.recording) return;
      try {
        const repo = encodeURIComponent(state.dataset.repoId);
        const payload = await api(`/api/datasets?repo_id=${repo}`, { method: "DELETE" });
        applyDatasets(payload);
        applySnapshot(payload);
        state.dataset.episode = null;
        await loadEpisodes();
        state.status = "Deleted";
      } catch (error) {
        state.status = error.message;
      }
    }

    function selectEpisode(item) {
      if (
        state.dataset.episode &&
        ((item.id && state.dataset.episode.id === item.id) ||
          (!item.id && state.dataset.episode.index === item.index))
      ) {
        state.dataset.episode = null;
      } else {
        state.dataset.episode = { ...item };
      }
    }

    async function saveEpisodeTask() {
      if (state.dataset.episode == null) return;
      try {
        const target = encodeURIComponent(
          state.dataset.episode.id || state.dataset.episode.index
        );
        const payload = await api(`/api/episodes/${target}`, {
          method: "PUT",
          body: JSON.stringify({
            task: state.dataset.episode.task,
            repo_id: state.dataset.repoId,
          }),
        });
        applyDatasets(payload);
        applySnapshot(payload);
        state.status = "Saved";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function deleteEpisode(target) {
      try {
        const repo = encodeURIComponent(state.dataset.repoId);
        const identifier = encodeURIComponent(target);
        const payload = await api(`/api/episodes/${identifier}?repo_id=${repo}`, {
          method: "DELETE",
        });
        applyDatasets(payload);
        applySnapshot(payload);
        if (
          state.dataset.episode?.id === target ||
          state.dataset.episode?.index === target
        ) {
          state.dataset.episode = null;
        }
        state.status = "Deleted";
      } catch (error) {
        state.status = error.message;
      }
    }

    function formatTrainEta(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value) || value < 0) return "";
      const total = Math.round(value);
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const secs = total % 60;
      if (hours) return `${hours}h ${minutes}m`;
      if (minutes) return `${minutes}m ${secs}s`;
      return `${secs}s`;
    }

    function formatTrainValue(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "—";
      if (Math.abs(number) > 0 && Math.abs(number) < 0.001) return number.toExponential(2);
      if (Math.abs(number) >= 100) return number.toFixed(1);
      return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    }

    function chartPoints(series) {
      if (!Array.isArray(series) || !series.length) return "";
      const width = 240;
      const height = 72;
      const xs = series.map((point) => Number(point.step));
      const ys = series.map((point) => Number(point.value));
      if (xs.some((x) => !Number.isFinite(x)) || ys.some((y) => !Number.isFinite(y))) return "";
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const dx = maxX - minX || 1;
      const dy = maxY - minY || 1;
      return series
        .map((_, index) => {
          const x = ((xs[index] - minX) / dx) * width;
          const y = height - ((ys[index] - minY) / dy) * height;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
    }

    function formatRunTime(stamp) {
      const date = new Date(Number(stamp) * 1000);
      if (Number.isNaN(date.getTime())) return "";
      return date.toLocaleString();
    }

    function runBusy(id) {
      return Boolean(state.runs.items.find((item) => item.id === id)?.busy);
    }

    function runCharts() {
      const history = state.runs.detail?.history || {};
      const kind = state.runs.detail?.kind;
      const charts = kind === "grpo"
        ? [
            { id: "loss", title: "Loss", series: history.loss },
            { id: "reward", title: "Reward", series: history.reward },
          ]
        : [
            { id: "loss", title: "Loss", series: history.loss },
            { id: "lr", title: "LR", series: history.lr },
          ];
      return charts.map((chart) => {
        const series = Array.isArray(chart.series) ? chart.series : [];
        return {
          ...chart,
          points: chartPoints(series),
          last: series.length ? series[series.length - 1].value : null,
        };
      });
    }

    function applyRuns(items) {
      if (!Array.isArray(items)) return;
      state.runs.items = items;
      if (!items.length) {
        state.runs.selected = "";
        state.runs.detail = null;
        return;
      }
      if (!items.some((item) => item.id === state.runs.selected)) {
        state.runs.selected = items[0].id;
      }
    }

    async function loadRuns() {
      try {
        const payload = await api("/api/runs");
        applyRuns(payload.runs);
        if (state.runs.selected) await selectRun(state.runs.selected);
      } catch (error) {
        state.status = error.message;
      }
    }

    async function selectRun(id) {
      state.runs.selected = id;
      try {
        const payload = await api(`/api/runs/${encodeURIComponent(id)}`);
        state.runs.detail = payload.run;
      } catch (error) {
        state.runs.detail = null;
        state.status = error.message;
      }
    }

    async function deleteRun(id) {
      if (!id || runBusy(id)) return;
      try {
        const payload = await api(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
        applyRuns(payload.runs);
        applySnapshot(payload);
        if (state.runs.selected) await selectRun(state.runs.selected);
        else state.runs.detail = null;
        loadCheckpoints();
        state.status = "Deleted";
      } catch (error) {
        state.status = error.message;
      }
    }

    function applyInferDefaults(config) {
      if (!config || state.test.running) return;
      const duration = config.environment?.rollout?.duration_seconds;
      if (duration != null) state.test.duration = duration;
      const hz = config.environment?.rollout?.control_hz;
      if (hz != null) state.test.controlHz = hz;
    }

    function applyEvalDefaults(config) {
      const ev = config?.environment?.eval;
      if (!ev || state.eval.running || state.eval.building) return;
      if (ev.size) state.eval.size = ev.size;
      if (ev.seed != null) state.eval.seed = ev.seed;
      if (ev.duration_seconds != null) state.eval.duration = ev.duration_seconds;
      if (ev.parallel) state.eval.parallel = ev.parallel;
      if (ev.denoising_steps) state.eval.numSteps = ev.denoising_steps;
      if (ev.n_action_steps) state.eval.nActionSteps = ev.n_action_steps;
    }

    function evalScenes() {
      return Array.isArray(state.eval.dataset?.scenes) ? state.eval.dataset.scenes : [];
    }

    function evalSceneResult(index) {
      return (state.eval.results || []).find((item) => item.index === index) || null;
    }

    function evalSelected() {
      const index = state.eval.selected;
      if (index == null) return null;
      return evalScenes().find((item) => item.index === index) || null;
    }

    function evalSceneImage(item, camera) {
      if (!item) return "";
      return `/api/eval/scene/${item.index}/image?camera=${encodeURIComponent(camera)}&t=${item.seed}`;
    }

    function selectEvalScene(index) {
      state.eval.selected = index;
      const item = evalScenes().find((row) => row.index === index);
      if (item) state.eval.rerollSeed = item.seed;
    }

    async function buildEvalSet() {
      if (isAct()) {
        state.status = "Eval is not available in ACT mode";
        return;
      }
      try {
        const payload = await api("/api/eval/dataset", {
          method: "POST",
          timeout: API_LONG_MS,
          body: JSON.stringify({
            size: Math.max(1, Number(state.eval.size) || 32),
            seed: Number(state.eval.seed) || 10000,
          }),
        });
        applySnapshot(payload, { drafts: true });
        state.status = "Building valset…";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function rerollEvalScene() {
      const item = evalSelected();
      if (!item) return;
      try {
        const payload = await api("/api/eval/dataset/reroll", {
          method: "POST",
          timeout: API_LONG_MS,
          body: JSON.stringify({
            index: item.index,
            seed: Number(state.eval.rerollSeed),
          }),
        });
        applySnapshot(payload, { drafts: true });
        state.status = "Rerolled";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function startEval() {
      if (isAct()) {
        state.status = "Eval is not available in ACT mode";
        return;
      }
      try {
        const payload = await api("/api/eval", {
          method: "POST",
          body: JSON.stringify({
            checkpoint: state.eval.checkpoint,
            policy_mode: "smolvla",
            duration_seconds: Number(state.eval.duration) || 40,
            parallel: Math.max(1, Number(state.eval.parallel) || 1),
            num_steps: Number(state.eval.numSteps) || 10,
            n_action_steps: Number(state.eval.nActionSteps) || 50,
          }),
        });
        applySnapshot(payload, { drafts: true });
        state.status = "Eval…";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function stopEval() {
      try {
        const payload = await api("/api/eval/stop", {
          method: "POST",
          body: "{}",
        });
        applySnapshot(payload);
        state.status = state.eval.running || state.eval.building ? "Stopping…" : "Stopped";
      } catch (error) {
        state.status = error.message;
      }
    }

    function applyGrpoDefaults(config) {
      const grpo = config?.environment?.grpo;
      if (!grpo || state.grpo.running) return;
      if (grpo.train_scope) state.grpo.trainScope = grpo.train_scope;
      if (grpo.group_size) state.grpo.groupSize = grpo.group_size;
      if (grpo.parallel_rollouts) state.grpo.parallel = grpo.parallel_rollouts;
      const duration = config?.environment?.rollout?.duration_seconds;
      if (duration != null) state.grpo.duration = duration;
      if (grpo.scenes_per_update) state.grpo.scenesPerUpdate = grpo.scenes_per_update;
      if (grpo.scene_waves) {
        state.grpo.sceneWaves = Math.min(
          Number(grpo.scene_waves) || 1,
          Number(state.grpo.scenesPerUpdate) || 1,
        );
      }
      if (grpo.flow?.sde_mode) state.grpo.sdeMode = grpo.flow.sde_mode;
      if (grpo.flow?.noise_level != null) state.grpo.noiseLevel = grpo.flow.noise_level;
      if (grpo.flow?.denoising_steps) state.grpo.denoisingSteps = grpo.flow.denoising_steps;
      if (grpo.optimization?.action_expert_learning_rate != null) {
        state.grpo.expertLr = grpo.optimization.action_expert_learning_rate;
      }
    }

    function grpoCharts() {
      const history = state.grpo.history || {};
      return [
        { id: "reward", title: "Reward", series: history.reward },
        { id: "success", title: "Success", series: history.success },
        { id: "loss", title: "Loss", series: history.loss },
        { id: "clip", title: "Clip", series: history.clip },
        { id: "drift", title: "Drift", series: history.drift },
        { id: "grad", title: "Grad", series: history.gradNorm },
      ].map((chart) => {
        const series = Array.isArray(chart.series) ? chart.series : [];
        return {
          ...chart,
          points: chartPoints(series),
          last: series.length ? series[series.length - 1].value : null,
        };
      });
    }

    function rewardBar(value) {
      const rewards = state.grpo.group.map((item) => Number(item.total_reward) || 0);
      const max = Math.max(0.01, ...rewards);
      const current = Number(value) || 0;
      return Math.max(0, Math.min(100, (current / max) * 100));
    }

    async function startGrpo() {
      if (isAct()) {
        state.status = "GRPO is not available in ACT mode";
        return;
      }
      try {
        const groupSize = Math.max(2, Number(state.grpo.groupSize) || 16);
        const parallel = Math.min(Math.max(1, Number(state.grpo.parallel) || 1), groupSize);
        const scenesPerUpdate = Math.max(1, Number(state.grpo.scenesPerUpdate) || 16);
        const sceneWaves = Math.min(
          Math.max(1, Number(state.grpo.sceneWaves) || 1),
          scenesPerUpdate,
        );
        state.grpo.groupSize = groupSize;
        state.grpo.parallel = parallel;
        state.grpo.scenesPerUpdate = scenesPerUpdate;
        state.grpo.sceneWaves = sceneWaves;
        const payload = await api("/api/grpo", {
          method: "POST",
          body: JSON.stringify({
            checkpoint: state.grpo.checkpoint,
            run: state.grpo.run,
            train_scope: state.grpo.trainScope,
            policy_mode: "smolvla",
            group_size: groupSize,
            parallel_rollouts: parallel,
            duration_seconds: Number(state.grpo.duration) || 40,
            scenes_per_update: scenesPerUpdate,
            scene_waves: sceneWaves,
            sde_mode: state.grpo.sdeMode,
            noise_level: Number(state.grpo.noiseLevel) || 0.35,
            denoising_steps: Number(state.grpo.denoisingSteps) || 10,
            expert_lr: Number(state.grpo.expertLr) || 5e-6,
          }),
        });
        applySnapshot(payload, { drafts: true });
        state.status = "GRPO…";
        if (state.grpo.scene) nextTick(() => refreshViews("grpo"));
      } catch (error) {
        state.status = error.message;
      }
    }

    async function stopGrpo() {
      try {
        const payload = await api("/api/grpo/stop", {
          method: "POST",
          body: "{}",
        });
        applySnapshot(payload);
        state.status = state.grpo.running ? "Stopping…" : "Stopped";
      } catch (error) {
        state.status = error.message;
      }
    }

    function trainCharts() {
      const history = state.train.history || {};
      return [
        { id: "loss", title: "Loss", series: history.loss },
        { id: "lr", title: "LR", series: history.lr },
        { id: "grad", title: "Grad", series: history.gradNorm },
      ].map((chart) => {
        const series = Array.isArray(chart.series) ? chart.series : [];
        return {
          ...chart,
          points: chartPoints(series),
          last: series.length ? series[series.length - 1].value : null,
        };
      });
    }

    function applyCheckpoints(items) {
      if (!Array.isArray(items)) return;
      state.test.allCheckpoints = items;
      const mode = isAct() ? "act" : "smolvla";
      const visible = items.filter((item) => (item.type || "smolvla") === mode);
      state.test.checkpoints = visible;
      if (!visible.some((item) => item.id === state.test.checkpoint)) {
        state.test.checkpoint = visible[0]?.id || "";
      }
      if (!visible.some((item) => item.id === state.grpo.checkpoint)) {
        state.grpo.checkpoint = visible[0]?.id || "";
      }
      if (!visible.some((item) => item.id === state.eval.checkpoint)) {
        state.eval.checkpoint = visible[0]?.id || "";
      }
    }

    async function loadCheckpoints() {
      try {
        const payload = await api("/api/checkpoints");
        applyCheckpoints(payload.checkpoints);
      } catch {
        return;
      }
    }

    async function startTrain() {
      try {
        const act = isAct() ? resolveActStart() : null;
        if (act) {
          state.train.policy = act.policy;
          state.train.run = act.run;
        }
        const policy = act ? act.policy : trainPolicyValue();
        const payload = await api("/api/train", {
          method: "POST",
          body: JSON.stringify({
            policy,
            policy_mode: isAct() ? "act" : "smolvla",
            epochs: Number(state.train.epochs) || 20,
            batch: Number(state.train.batch) || DEFAULT_TRAIN_BATCH,
            lr: Number(state.train.lr) || 0.0001,
            save_every: Number(state.train.saveEvery) || 0,
            run: act ? act.run : state.train.run,
            tune: state.train.tune,
            repo_ids: state.train.repoIds,
          }),
        });
        applySnapshot(payload, { drafts: true });
        state.status = "Training…";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function stopTrain() {
      try {
        const payload = await api("/api/train/stop", {
          method: "POST",
          body: "{}",
        });
        applySnapshot(payload);
        state.status = state.train.running ? "Stopping…" : "Stopped";
      } catch (error) {
        state.status = error.message;
      }
    }

    async function runTest() {
      if (state.test.running || state.infer.building || !state.infer.scene) return;
      state.test.running = true;
      state.stats = { frames: 0, hz: 0 };
      overlayInfer("");
      state.status = "Starting test…";
      addLog("Starting test…");
      try {
        const payload = await api("/api/test", {
          method: "POST",
          body: JSON.stringify({
            duration_seconds: Number(state.test.duration) || 20,
            n_action_steps: Number(state.test.nActionSteps) || 50,
            num_steps: Number(state.test.numSteps) || 10,
            control_hz: Number(state.test.controlHz) || 15,
            fps: Number(state.test.viewFps) || 15,
            checkpoint: state.test.checkpoint,
            policy_mode: isAct() ? "act" : "smolvla",
          }),
        });
        applySnapshot(payload);
      } catch (error) {
        if (!isNetworkError(error)) {
          testWasRunning = false;
          state.test.running = false;
          overlayInfer(error.message, true);
          addLog(error.message, "error");
          state.status = "Failed";
          await refreshViews("inference");
          return;
        }
      }
      if (state.test.running) startInferViews();
    }

    async function stopTest() {
      testWasRunning = false;
      try {
        const payload = await api("/api/test/stop", { method: "POST", body: "{}" });
        applySnapshot(payload);
      } catch (error) {
        if (!isNetworkError(error)) state.status = error.message;
      }
      if (state.test.running) return;
      stopInferViews();
      clearCameraSrcs("inference");
      addLog("Stopped.");
      state.status = "Stopped";
      overlayInfer("");
      if (state.infer.scene) await refreshViews("inference");
    }

    function showPage() {
      const previous = state.page;
      state.page = pageFromHash();
      if (isAct() && (state.page === "grpo" || state.page === "eval")) {
        location.hash = "train";
        return;
      }
      if (previous === "collect" && state.page !== "collect") stopCollectViews();
      if (previous === "inference" && state.page !== "inference") {
        stopInferViews();
        clearCameraSrcs("inference");
      }
      if (previous === "grpo" && state.page !== "grpo") {
        stopGrpoViews();
        clearCameraSrcs("grpo");
      }
      if (state.page === "collect" && state.collect.step === "ready") {
        nextTick(() => {
          scheduleLayout("collect");
          if (state.collect.scene) startCollectViews();
        });
      }
      if (state.page === "inference" && state.infer.scene) {
        nextTick(() => refreshViews("inference"));
      }
      if (state.page === "datasets") loadDatasets();
      if (state.page === "train" || state.page === "collect") loadDatasets({ skipEpisodes: true });
      if (state.page === "grpo") {
        loadCheckpoints();
        if (state.grpo.scene) nextTick(() => refreshViews("grpo"));
      }
      if (state.page === "eval") loadCheckpoints();
      if (state.page === "inference") loadCheckpoints();
      if (state.page === "runs") loadRuns();
      syncSlotOverlays();
    }

    onMounted(async () => {
      SETTINGS_NAV.forEach((item) => {
        state.openGroup[item.id] = groupsFor(item.id)[0]?.id || "";
      });
      window.addEventListener("hashchange", showPage);
      const collectStage = document.getElementById("collect-stage");
      const inferStage = document.getElementById("infer-stage");
      const grpoStage = document.getElementById("grpo-stage");
      if (typeof ResizeObserver !== "undefined") {
        if (collectStage) new ResizeObserver(() => applyLayout("collect")).observe(collectStage);
        if (inferStage) new ResizeObserver(() => applyLayout("inference")).observe(inferStage);
        if (grpoStage) new ResizeObserver(() => applyLayout("grpo")).observe(grpoStage);
      } else {
        window.addEventListener("resize", () => {
          applyLayout("collect");
          applyLayout("inference");
          applyLayout("grpo");
        });
      }
      try {
        const payload = await api("/api/state?full=1");
        applySnapshot(payload, { boot: true });
        if (isAct() && (state.section === "grpo" || state.section === "eval")) state.section = "training";
        syncActTrainForm();
        if (state.collect.record.repoId) state.dataset.repoId = state.collect.record.repoId;
        if (
          !state.train.running
          && !state.grpo.running
          && !state.eval.running
          && !state.eval.building
          && !state.test.running
          && !state.collect.record.recording
        ) {
          state.status = "Ready";
        }
      } catch (error) {
        state.status = error.message;
      }
      showPage();
      scheduleStatePoll();
      if (state.page === "collect" && state.collect.step === "ready" && state.collect.scene) {
        await refreshViews("collect");
      }
      if (state.infer.scene) {
        overlayInfer("");
        if (state.page === "inference") await refreshViews("inference");
      }
    });

    return {
      SETTINGS_NAV,
      settingsNav,
      state,
      statusText,
      isAct,
      setPolicyMode,
      trainPolicies,
      trainPolicySelect,
      setTrainPolicySelect,
      isCustomTrainPolicy,
      trainPolicyReady,
      groupsFor,
      isOpen,
      toggleGroup,
      fieldValue,
      commit,
      commitNumber,
      updateIndex,
      vec3List,
      addVec3,
      removeVec3,
      updateVec3List,
      catalogItems,
      catalogPreview,
      filteredCatalog,
      isCatalogChecked,
      catalogSelectedCount,
      setCatalogChecked,
      selectCatalogRole,
      objectLabel,
      setObjectLabel,
      collectCameraNames,
      inferCameraNames,
      policyMapSlots,
      policyMapSlot,
      policyMapChoices,
      setPolicyMapSlot,
      grpoCameraNames,
      collectTitle,
      wizardLabel,
      wizardNext,
      regenerate,
      resetScene,
      generateScene,
      runTest,
      stopTest,
      detectLead,
      detectUnplug,
      detectReplug,
      connectLead,
      useSavedCalibration,
      startCalibrate,
      setHome,
      finishCalibrate,
      saveCollectSettings,
      startRecord,
      stopRecord,
      episodeVideo,
      datasetStats,
      selectCollectDataset,
      selectDataset,
      createDataset,
      deleteDataset,
      importHubDataset,
      syncHubDataset,
      isHubDataset,
      selectEpisode,
      saveEpisodeTask,
      deleteEpisode,
      startTrain,
      stopTrain,
      startGrpo,
      stopGrpo,
      evalScenes,
      evalSceneResult,
      evalSelected,
      evalSceneImage,
      selectEvalScene,
      buildEvalSet,
      rerollEvalScene,
      startEval,
      stopEval,
      formatTrainEta,
      formatTrainValue,
      trainCharts,
      grpoCharts,
      runCharts,
      formatRunTime,
      runBusy,
      selectRun,
      deleteRun,
      rewardBar,
    };
  },
}).mount("#app");

async function api(path, options = {}) {
  const { timeout = API_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...fetchOptions,
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Request timed out");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
