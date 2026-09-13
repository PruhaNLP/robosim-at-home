from __future__ import annotations

import threading
import time

from core.config import normalize_policy_mode
from devices import apply_render_device, resolve_model_device, resolve_render_device
from eval.dataset import build_valset, load_manifest, reroll_scene

# ======Settings=========
DEFAULT_SIZE = 32
DEFAULT_SEED = 10000
DEFAULT_DURATION = 40
DEFAULT_PARALLEL = 8
DEFAULT_DENOISE = 10
DEFAULT_N_ACTION = 50
LOG_LIMIT = 400
# ======Settings=========


def _idle_progress() -> dict:
    return {
        "phase": "idle",
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "etaSeconds": None,
        "label": "",
    }


class EvalRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.running = False
        self.building = False
        self.error: str | None = None
        self.logs: list[dict] = []
        self.checkpoint = ""
        self.size = DEFAULT_SIZE
        self.seed = DEFAULT_SEED
        self.duration = DEFAULT_DURATION
        self.parallel = DEFAULT_PARALLEL
        self.num_steps = DEFAULT_DENOISE
        self.n_action_steps = DEFAULT_N_ACTION
        self.progress = _idle_progress()
        self.dataset = load_manifest()
        if self.dataset.get("size"):
            self.size = int(self.dataset["size"])
        if self.dataset.get("seed") is not None and self.dataset.get("scenes"):
            self.seed = int(self.dataset["seed"])
        self.results: list[dict] = []
        self.metrics: dict | None = None
        self._progress_t0 = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "building": self.building,
                "checkpoint": self.checkpoint,
                "size": self.size,
                "seed": self.seed,
                "duration": self.duration,
                "parallel": self.parallel,
                "numSteps": self.num_steps,
                "nActionSteps": self.n_action_steps,
                "error": self.error,
                "logs": list(self.logs[-LOG_LIMIT:]),
                "progress": dict(self.progress),
                "dataset": dict(self.dataset),
                "results": list(self.results),
                "metrics": dict(self.metrics) if self.metrics else None,
            }

    def log(self, text: str, kind: str = "info") -> None:
        line = str(text or "").strip()
        if not line:
            return
        with self.lock:
            self.logs.append({"kind": kind, "text": line})
            self.logs = self.logs[-LOG_LIMIT:]

    def _set_progress(self, phase: str, current: int, total, label: str) -> None:
        total_n = 0 if total in (None, "") else int(total)
        percent = 0.0
        if total_n > 0:
            percent = min(100.0, round(100.0 * current / total_n, 1))
        with self.lock:
            if phase != self.progress.get("phase"):
                self._progress_t0 = time.monotonic()
            eta = None
            if total_n > current >= 1:
                elapsed = time.monotonic() - self._progress_t0
                if elapsed > 0:
                    eta = round(elapsed / current * (total_n - current))
            self.progress.update(
                {
                    "phase": phase,
                    "current": int(current),
                    "total": total_n,
                    "percent": percent,
                    "etaSeconds": eta,
                    "label": label,
                }
            )

    def _busy(self) -> bool:
        return self.running or self.building

    def start_build(self, payload: dict, compute: dict, worker=None) -> dict:
        with self.lock:
            if self._busy():
                raise RuntimeError("eval is already busy")
            self.size = max(1, int(payload.get("size") or DEFAULT_SIZE))
            self.seed = int(payload.get("seed") or DEFAULT_SEED)
            self.error = None
            self.logs = []
            self.results = []
            self.metrics = None
            self.progress = _idle_progress()
            self.stop.clear()
            self.building = True
            self._progress_t0 = time.monotonic()
        self.log(f"Build valset · {self.size} scenes · seed {self.seed}.")
        self.thread = threading.Thread(
            target=self._build,
            args=(compute, worker),
            daemon=True,
        )
        self.thread.start()
        return self.snapshot()

    def reroll(self, payload: dict, compute: dict, worker=None) -> dict:
        with self.lock:
            if self._busy():
                raise RuntimeError("eval is already busy")
            self.error = None
        index = int(payload.get("index"))
        seed = payload.get("seed")
        seed_n = None if seed in (None, "") else int(seed)
        if worker is not None:
            worker.close()
        render = resolve_render_device(compute)
        apply_render_device(render)
        self.log(f"Reroll scene {index + 1}" + (f" seed {seed_n}." if seed_n is not None else "."))
        manifest = reroll_scene(index, seed_n)
        with self.lock:
            self.dataset = manifest
            self.size = int(manifest.get("size") or self.size)
            self.results = [item for item in self.results if int(item.get("index")) != index]
            if self.metrics:
                n_ok = sum(1 for item in self.results if item.get("success"))
                self.metrics = {
                    **self.metrics,
                    "n_scenes": len(self.results),
                    "n_success": n_ok,
                    "success_rate": n_ok / max(1, len(self.results)),
                }
        scene = next(
            (item for item in manifest.get("scenes") or [] if int(item["index"]) == index),
            None,
        )
        if scene:
            self.log(f"Scene {index + 1} · seed {scene['seed']} · {scene.get('target') or ''}.", "ok")
        return self.snapshot()

    def start(self, payload: dict, compute: dict, config: dict, worker=None) -> dict:
        mode = normalize_policy_mode(payload.get("policy_mode") or config.get("policy_mode"))
        if mode == "act":
            raise RuntimeError("Eval batch is not available in ACT mode.")
        with self.lock:
            if self._busy():
                raise RuntimeError("eval is already busy")
            self.checkpoint = str(payload.get("checkpoint") or "").strip()
            if not self.checkpoint:
                raise RuntimeError("select a checkpoint")
            self.duration = max(1.0, float(payload.get("duration_seconds") or DEFAULT_DURATION))
            self.parallel = max(1, int(payload.get("parallel") or DEFAULT_PARALLEL))
            self.num_steps = max(1, int(payload.get("num_steps") or DEFAULT_DENOISE))
            self.n_action_steps = max(1, int(payload.get("n_action_steps") or DEFAULT_N_ACTION))
            self.error = None
            self.logs = []
            self.results = []
            self.metrics = None
            self.progress = _idle_progress()
            self.dataset = load_manifest()
            self.stop.clear()
            self.running = True
            self._progress_t0 = time.monotonic()
        if not (self.dataset.get("scenes") or []):
            with self.lock:
                self.running = False
            raise RuntimeError("build a validation set first")
        self.log(f"Eval {self.checkpoint}.")
        self.thread = threading.Thread(
            target=self._run,
            args=(compute, config, worker),
            daemon=True,
        )
        self.thread.start()
        return self.snapshot()

    def stop_run(self) -> dict:
        self.stop.set()
        self.log("Stopping…", "warn")
        return self.snapshot()

    def _build(self, compute: dict, worker) -> None:
        try:
            if worker is not None:
                worker.close()
            render = resolve_render_device(compute)
            apply_render_device(render)
            manifest = build_valset(
                self.size,
                self.seed,
                self.stop.is_set,
                on_progress=self._set_progress,
                on_log=self.log,
            )
            with self.lock:
                self.dataset = manifest
                self.size = int(manifest.get("size") or self.size)
            if self.stop.is_set():
                self.log("Build stopped.", "warn")
            else:
                self.log(f"Valset ready · {len(manifest.get('scenes') or [])} scenes.", "ok")
        except Exception as error:
            self.log(str(error), "error")
            with self.lock:
                self.error = str(error)
                self.progress["phase"] = "error"
                self.progress["label"] = str(error)
        finally:
            with self.lock:
                self.building = False

    def _run(self, compute: dict, config: dict, worker) -> None:
        try:
            if worker is not None:
                worker.close()
            render = resolve_render_device(compute)
            apply_render_device(render)
            device_name = resolve_model_device(compute.get("model"))
            import torch
            from eval.run import EvalHooks, run_eval

            def on_result(row: dict) -> None:
                with self.lock:
                    self.results.append(row)

            def on_metrics(info: dict) -> None:
                with self.lock:
                    self.metrics = dict(info)

            run_eval(
                config,
                self.checkpoint,
                torch.device(device_name),
                EvalHooks(
                    should_stop=self.stop.is_set,
                    log=self.log,
                    progress=self._set_progress,
                    result=on_result,
                    metrics=on_metrics,
                ),
                self.duration,
                self.parallel,
                self.n_action_steps,
                self.num_steps,
            )
        except Exception as error:
            self.log(str(error), "error")
            with self.lock:
                self.error = str(error)
                self.progress["phase"] = "error"
                self.progress["label"] = str(error)
        finally:
            with self.lock:
                self.running = False
