from __future__ import annotations

import json
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

# ======Settings=========
DEFAULT_REPO_ID = "local/so100_collect"
EPISODE_PREFIX = "episode_"
DATASET_META = "dataset.json"
TRAJ_NAME = "traj.npz"
TRAJECTORY_SCHEMA_VERSION = 2
ACTION_SEMANTICS = "simulator_absolute_joint_targets_degrees"
STATE_SEMANTICS = "simulator_joint_positions_degrees"
ACTION_DIM = 6
CAMERA_ORDER = ("front", "wrist")
HUB_SOURCE = "hub"
HUB_TMP_SUFFIX = ".__hub"
FFMPEG_BIN = "ffmpeg"
VIDEO_TS_TOLERANCE = 1e-4
LOG_LIMIT = 200
# ======Settings=========


def dataset_root(base: Path, repo_id: str) -> Path:
    return base / repo_id.replace("/", "_")


def _episode_index(name: str) -> int | None:
    if not name.startswith(EPISODE_PREFIX):
        return None
    try:
        return int(name[len(EPISODE_PREFIX) :])
    except ValueError:
        return None


class DatasetStore:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.repo_id = DEFAULT_REPO_ID
        self.current: Path | None = None
        existing = self._existing_repo_ids()
        if existing and self.repo_id not in existing:
            self.repo_id = existing[0]

    def _existing_repo_ids(self) -> list[str]:
        if not self.base.is_dir():
            return []
        ids = []
        for path in sorted(self.base.iterdir()):
            if not path.is_dir() or path.name.endswith(HUB_TMP_SUFFIX):
                continue
            ids.append(self._repo_from_folder(path))
        return ids

    def root(self, repo_id: str | None = None) -> Path:
        return dataset_root(self.base, repo_id or self.repo_id)

    def current_index(self) -> int:
        if self.current is None:
            return 0
        return _episode_index(self.current.name) or 0

    def is_current(self, target: str | int, repo_id: str | None = None) -> bool:
        folder, _ = self._find_episode_folder(target, repo_id)
        return folder is not None and folder == self.current

    def next_index(self, repo_id: str | None = None) -> int:
        root = self.root(repo_id)
        nums = []
        if root.is_dir():
            for path in root.iterdir():
                index = _episode_index(path.name) if path.is_dir() else None
                if index is not None:
                    nums.append(index)
        return (max(nums) + 1) if nums else 1

    def _read_dataset_meta(self, repo_id: str | None = None) -> dict:
        path = self.root(repo_id) / DATASET_META
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_dataset_meta(self, repo_id: str, extra: dict | None = None) -> None:
        root = self.root(repo_id)
        root.mkdir(parents=True, exist_ok=True)
        payload = self._read_dataset_meta(repo_id)
        payload["repo_id"] = repo_id
        if extra:
            payload.update(extra)
        (root / DATASET_META).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def is_hub(self, repo_id: str | None = None) -> bool:
        return str(self._read_dataset_meta(repo_id).get("source") or "") == HUB_SOURCE

    def _repo_from_folder(self, path: Path) -> str:
        meta_path = path / DATASET_META
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            repo_id = str(meta.get("repo_id") or "").strip()
            if repo_id:
                return repo_id
        name = path.name
        if "_" in name:
            return name.replace("_", "/", 1)
        return name

    def list_datasets(self) -> list[dict]:
        items = []
        seen = set()
        if self.base.is_dir():
            for path in sorted(self.base.iterdir()):
                if not path.is_dir() or path.name.endswith(HUB_TMP_SUFFIX):
                    continue
                repo_id = self._repo_from_folder(path)
                seen.add(repo_id)
                episodes = self.list_episodes(repo_id)
                meta = self._read_dataset_meta(repo_id)
                items.append(
                    {
                        "repoId": repo_id,
                        "episodes": len(episodes),
                        "trainable": sum(1 for item in episodes if item.get("trainable")),
                        "source": str(meta.get("source") or "local"),
                        "hubRepo": str(meta.get("hub_repo") or "") or None,
                    }
                )
        if self.repo_id not in seen:
            items.insert(
                0,
                {
                    "repoId": self.repo_id,
                    "episodes": 0,
                    "trainable": 0,
                    "source": "local",
                    "hubRepo": None,
                },
            )
        return items

    def create_dataset(self, repo_id: str) -> str:
        repo_id = str(repo_id or "").strip()
        if not repo_id:
            raise ValueError("empty repo id")
        self._write_dataset_meta(repo_id)
        self.repo_id = repo_id
        return repo_id

    def delete_dataset(self, repo_id: str) -> None:
        root = self.root(repo_id)
        if self.current is not None and self.current.is_relative_to(root):
            self.current = None
        if root.exists():
            shutil.rmtree(root)
        if self.repo_id == repo_id:
            self.repo_id = DEFAULT_REPO_ID

    def _find_episode_folder(
        self, target: str | int, repo_id: str | None = None
    ) -> tuple[Path | None, int | None]:
        root = self.root(repo_id)
        if not root.is_dir():
            return None, None
        target_str = str(target).strip()
        if target_str.isdigit():
            idx = int(target_str)
            folder = root / f"{EPISODE_PREFIX}{idx}"
            if folder.is_dir():
                return folder, idx
        for folder in sorted(root.iterdir(), key=lambda path: path.name):
            if not folder.is_dir():
                continue
            idx = _episode_index(folder.name)
            if idx is None:
                continue
            meta_path = folder / "meta.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                    if (
                        str(meta.get("id")) == target_str
                        or str(meta.get("index")) == target_str
                    ):
                        return folder, idx
                except Exception:
                    pass
        return None, None

    def _recover_tmp_episodes(self, repo_id: str | None = None) -> None:
        root = self.root(repo_id)
        if not root.is_dir():
            return
        used = set()
        leftovers = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            idx = _episode_index(path.name)
            if idx is not None:
                used.add(idx)
            elif path.name.startswith("_tmp_"):
                leftovers.append(path)
        leftovers.sort(key=lambda path: path.stat().st_mtime)
        for path in leftovers:
            meta_path = path / "meta.json"
            meta = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = {}
            wanted = int(meta.get("index") or 0)
            if wanted < 1 or wanted in used:
                wanted = 1
                while wanted in used:
                    wanted += 1
            target = root / f"{EPISODE_PREFIX}{wanted}"
            path.rename(target)
            used.add(wanted)
            meta["index"] = wanted
            if not meta.get("id"):
                meta["id"] = f"ep_{uuid.uuid4().hex[:8]}"
            (target / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )

    def _renumber_episodes(self, repo_id: str | None = None) -> None:
        root = self.root(repo_id)
        if not root.is_dir():
            return
        self._recover_tmp_episodes(repo_id)
        folder_list = []
        for path in root.iterdir():
            if path.is_dir():
                idx = _episode_index(path.name)
                if idx is not None:
                    folder_list.append((idx, path))
        folder_list.sort(key=lambda x: x[0])
        for new_idx, (old_idx, path) in enumerate(folder_list, start=1):
            target = path if old_idx == new_idx else root / f"{EPISODE_PREFIX}{new_idx}"
            if path != target:
                path.rename(target)
            meta_path = target / "meta.json"
            meta = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = {}
            meta["index"] = new_idx
            if not meta.get("id"):
                meta["id"] = f"ep_{uuid.uuid4().hex[:8]}"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def begin_episode(
        self,
        repo_id: str,
        task: str,
        fps: int,
        cameras: list[str],
    ) -> Path:
        if self.current is not None:
            self.discard_episode()
        self.repo_id = repo_id
        self._write_dataset_meta(repo_id)
        index = self.next_index(repo_id)
        ep_id = f"ep_{uuid.uuid4().hex[:8]}"
        folder = self.root(repo_id) / f"{EPISODE_PREFIX}{index}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "id": ep_id,
                    "index": index,
                    "task": task,
                    "fps": int(fps),
                    "cameras": list(cameras),
                    "frames": 0,
                    "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                    "action_semantics": ACTION_SEMANTICS,
                    "state_semantics": STATE_SEMANTICS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        self.current = folder
        return folder

    def finish_episode(self, frames: int) -> Path:
        folder = self.current
        if folder is None:
            raise RuntimeError("no episode in progress")
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta["frames"] = int(frames)
        if not meta.get("id"):
            meta["id"] = f"ep_{uuid.uuid4().hex[:8]}"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )
        self.current = None
        return folder

    def save_trajectory(self, actions: list, states: list) -> None:
        folder = self.current
        if folder is None:
            raise RuntimeError("no episode in progress")
        import numpy as np

        np.savez_compressed(
            folder / TRAJ_NAME,
            action=np.asarray(actions, dtype=np.float32),
            state=np.asarray(states, dtype=np.float32),
        )

    def discard_episode(self) -> None:
        folder = self.current
        self.current = None
        if folder is not None and folder.exists():
            shutil.rmtree(folder)
            self._renumber_episodes(self.repo_id)

    def list_episodes(self, repo_id: str | None = None) -> list[dict]:
        root = self.root(repo_id)
        if not root.is_dir():
            return []
        self._recover_tmp_episodes(repo_id)
        items = []
        for folder in sorted(root.iterdir(), key=lambda path: path.name):
            index = _episode_index(folder.name)
            if index is None or not folder.is_dir():
                continue
            meta_path = folder / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            ep_id = meta.get("id")
            if not ep_id:
                ep_id = f"ep_{uuid.uuid4().hex[:8]}"
                meta["id"] = ep_id
                meta["index"] = index
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            cameras = [
                path.stem
                for path in sorted(folder.glob("*.mp4"))
            ]
            if not cameras:
                cameras = list(meta.get("cameras") or [])
            fps = float(meta.get("fps") or 1)
            frames = int(meta.get("frames") or 0)
            items.append(
                {
                    "id": str(ep_id),
                    "index": index,
                    "task": str(meta.get("task") or ""),
                    "frames": frames,
                    "seconds": round(frames / fps, 2) if fps else 0.0,
                    "cameras": cameras,
                    "trainable": (folder / TRAJ_NAME).is_file(),
                }
            )
        return items

    def video_path(self, target: str | int, camera: str, repo_id: str | None = None) -> Path:
        folder, _ = self._find_episode_folder(target, repo_id)
        if folder is None or not folder.is_dir():
            raise FileNotFoundError(f"no video for {camera} episode {target}")
        path = folder / f"{camera}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"no video for {camera} episode {target}")
        return path

    def edit_task(self, target: str | int, task: str, repo_id: str | None = None) -> None:
        folder, _ = self._find_episode_folder(target, repo_id)
        if folder is None or not folder.is_dir():
            raise FileNotFoundError(f"episode {target} not found")
        meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta["task"] = str(task)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def delete_episode(self, target: str | int, repo_id: str | None = None) -> None:
        folder, _ = self._find_episode_folder(target, repo_id)
        if folder is not None and folder.exists():
            if self.current is not None and self.current == folder:
                self.current = None
            shutil.rmtree(folder)
        self._renumber_episodes(repo_id)


def _hub_id(repo_id: str) -> str:
    repo_id = str(repo_id or "").strip()
    if not repo_id or "/" not in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise RuntimeError("HF dataset must look like org/name")
    return repo_id


def _feature_dim(feat: dict) -> int:
    shape = feat.get("shape") or ()
    if shape:
        return int(shape[-1])
    names = feat.get("names") or []
    return len(names) if isinstance(names, list) else 0


def _hub_video_keys(meta) -> list[str]:
    depth = set(getattr(meta, "depth_keys", None) or [])
    keys = []
    for key in meta.camera_keys:
        if key in depth:
            continue
        if meta.features.get(key, {}).get("dtype") != "video":
            continue
        keys.append(key)
    return keys


def _camera_map(camera_keys: list[str]) -> dict[str, str]:
    names = [(key.rsplit(".", 1)[-1].lower(), key) for key in camera_keys]
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for want in CAMERA_ORDER:
        for short, key in names:
            if short == want and key not in used:
                mapping[want] = key
                used.add(key)
                break
    leftovers = [key for _, key in names if key not in used]
    for want in CAMERA_ORDER:
        if want not in mapping and leftovers:
            mapping[want] = leftovers.pop(0)
    return mapping


def validate_hub_dataset(repo_id: str) -> dict:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    repo_id = _hub_id(repo_id)
    meta = LeRobotDatasetMetadata(repo_id)
    errors = []
    if "observation.state" not in meta.features:
        errors.append("missing observation.state")
    elif _feature_dim(meta.features["observation.state"]) < ACTION_DIM:
        errors.append(
            f"observation.state dim {_feature_dim(meta.features['observation.state'])} < {ACTION_DIM}"
        )
    if "action" not in meta.features:
        errors.append("missing action")
    elif _feature_dim(meta.features["action"]) < ACTION_DIM:
        errors.append(
            f"action dim {_feature_dim(meta.features['action'])} < {ACTION_DIM}"
        )
    video_keys = _hub_video_keys(meta)
    if not video_keys:
        errors.append("no video cameras")
    if int(meta.total_episodes or 0) < 1:
        errors.append("no episodes")
    if int(meta.fps or 0) < 1:
        errors.append("invalid fps")
    if errors:
        raise RuntimeError(f"{repo_id}: " + "; ".join(errors))
    cam_map = _camera_map(video_keys)
    return {
        "repoId": repo_id,
        "episodes": int(meta.total_episodes),
        "frames": int(meta.total_frames),
        "fps": int(meta.fps),
        "cameras": list(cam_map.keys()),
        "cameraKeys": cam_map,
    }


def _stack_feature(rows: dict, name: str):
    import numpy as np

    if name not in rows:
        raise RuntimeError(f"missing feature {name}")
    raw = rows[name]
    if hasattr(raw, "numpy") and not isinstance(raw, np.ndarray):
        raw = raw.numpy()
    if isinstance(raw, np.ndarray) and raw.dtype != object and raw.ndim >= 2:
        return np.asarray(raw, dtype=np.float32)
    return np.stack([np.asarray(value, dtype=np.float32) for value in raw])


def _hub_timestamps(rows: dict) -> list[float]:
    import numpy as np

    if "timestamp" not in rows:
        raise RuntimeError("missing feature timestamp")
    out = []
    for value in rows["timestamp"]:
        item = np.asarray(value).reshape(-1)
        out.append(float(item[0]))
    return out


def _write_mp4(path: Path, frames, fps: int) -> None:
    import numpy as np

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise RuntimeError(f"need RGB video, got {array.shape}")
    height = int(array.shape[1])
    width = int(array.shape[2])
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        array = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
        height += pad_h
        width += pad_w
    proc = subprocess.Popen(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(int(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(np.ascontiguousarray(array, dtype=np.uint8).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr else b""
        code = proc.wait()
    except Exception:
        proc.kill()
        raise
    if code:
        text = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(text or f"cannot write {path.name}")


def _episode_task(dataset, start: int) -> str:
    import numpy as np

    row = dataset.hf_dataset[int(start)]
    if "task_index" not in row:
        return ""
    task_idx = int(np.asarray(row["task_index"]).reshape(-1)[0])
    try:
        return str(dataset.meta.tasks.iloc[task_idx].name)
    except Exception:
        return ""


def convert_hub_dataset(dataset, dest: Path, log, progress=None) -> int:
    import numpy as np
    from lerobot.datasets.video_utils import decode_video_frames

    dest.mkdir(parents=True, exist_ok=True)
    cam_map = _camera_map(_hub_video_keys(dataset.meta))
    if not cam_map:
        raise RuntimeError(f"{dataset.repo_id}: no video cameras")
    fps = int(dataset.meta.fps)
    if dataset.meta.episodes is None:
        raise RuntimeError(f"{dataset.repo_id}: no episodes")
    added = 0
    total = len(dataset.meta.episodes)
    for index in range(total):
        if progress is not None:
            progress("convert", index + 1, total, f"Convert #{index}")
        ep = dataset.meta.episodes[index]
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])
        length = end - start
        if length < 1:
            log(f"Skip #{index}: empty episode.", "warn")
            continue
        rows = dataset.hf_dataset[start:end]
        try:
            state = _stack_feature(rows, "observation.state")[:, :ACTION_DIM]
            action = _stack_feature(rows, "action")[:, :ACTION_DIM]
        except Exception as error:
            log(f"Skip #{index}: {error}", "warn")
            continue
        if state.shape[0] != length or action.shape[0] != length:
            log(f"Skip #{index}: trajectory length mismatch.", "warn")
            continue
        if state.shape[-1] < ACTION_DIM or action.shape[-1] < ACTION_DIM:
            log(f"Skip #{index}: state/action dim < {ACTION_DIM}.", "warn")
            continue
        timestamps = _hub_timestamps(rows)
        collect_index = added + 1
        folder = dest / f"{EPISODE_PREFIX}{collect_index}"
        folder.mkdir(parents=True, exist_ok=True)
        cameras = []
        try:
            for name, key in cam_map.items():
                if progress is not None:
                    progress(
                        "convert",
                        index + 1,
                        total,
                        f"Decode #{index} {name}",
                    )
                from_key = f"videos/{key}/from_timestamp"
                try:
                    from_ts = float(ep[from_key])
                except (KeyError, TypeError):
                    from_ts = 0.0
                shifted = [from_ts + stamp for stamp in timestamps]
                video_path = dataset.root / dataset.meta.get_video_file_path(index, key)
                frames = decode_video_frames(
                    video_path, shifted, VIDEO_TS_TOLERANCE, None, return_uint8=True
                )
                if frames.ndim != 4:
                    raise RuntimeError(f"bad video tensor {name}")
                if int(frames.shape[1]) == 3:
                    frames = frames.permute(0, 2, 3, 1).contiguous()
                array = frames.detach().cpu().numpy()
                if array.shape[0] != length:
                    raise RuntimeError(
                        f"{name}: {array.shape[0]} frames, trajectory {length}"
                    )
                _write_mp4(folder / f"{name}.mp4", array, fps)
                cameras.append(name)
        except Exception as error:
            shutil.rmtree(folder, ignore_errors=True)
            log(f"Skip #{index}: {error}", "warn")
            continue
        np.savez_compressed(folder / TRAJ_NAME, action=action, state=state)
        (folder / "meta.json").write_text(
            json.dumps(
                {
                    "id": f"ep_{uuid.uuid4().hex[:8]}",
                    "index": collect_index,
                    "task": _episode_task(dataset, start),
                    "fps": fps,
                    "cameras": cameras,
                    "frames": int(length),
                    "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                    "action_semantics": ACTION_SEMANTICS,
                    "state_semantics": STATE_SEMANTICS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        added += 1
        log(f"Added #{index} → {collect_index} · {length} frames.")
    if added < 1:
        raise RuntimeError(f"{dataset.repo_id}: no convertible episodes")
    return added


class HubImportRuntime:
    def __init__(self, store: DatasetStore) -> None:
        self.store = store
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.error: str | None = None
        self.repo_id: str | None = None
        self.logs: list[dict] = []
        self.progress = {
            "phase": "idle",
            "current": 0,
            "total": 0,
            "percent": 0.0,
            "label": "",
        }

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "repoId": self.repo_id,
                "logs": list(self.logs[-LOG_LIMIT:]),
                "progress": dict(self.progress),
            }

    def log(self, text: str, kind: str = "info") -> None:
        line = str(text or "").strip()
        if not line:
            return
        with self.lock:
            self.logs.append({"kind": kind, "text": line})
            self.logs = self.logs[-LOG_LIMIT:]

    def _set_progress(self, phase: str, current: int, total: int, label: str) -> None:
        percent = 0.0
        if total > 0:
            percent = min(100.0, round(100.0 * current / total, 1))
        with self.lock:
            self.progress.update(
                {
                    "phase": phase,
                    "current": int(current),
                    "total": int(total),
                    "percent": percent,
                    "label": label,
                }
            )

    def start(self, repo_id: str, replace: bool = False) -> dict:
        repo_id = _hub_id(repo_id)
        with self.lock:
            if self.running:
                raise RuntimeError("hub import already running")
        root = self.store.root(repo_id)
        if root.exists() and not replace:
            raise RuntimeError(f"{repo_id} already exists. Use Sync to replace it.")
        if replace and root.exists() and not self.store.is_hub(repo_id):
            raise RuntimeError(f"{repo_id} is a local dataset")
        info = validate_hub_dataset(repo_id)
        with self.lock:
            self.running = True
            self.error = None
            self.repo_id = repo_id
            self.logs = []
            self.progress = {
                "phase": "prepare",
                "current": 0,
                "total": 1,
                "percent": 0.0,
                "label": f"Check {repo_id}",
            }
        self.log(
            f"HF {repo_id} · {info['episodes']} episodes · {info['frames']} frames · "
            f"{', '.join(info['cameras'])} · {info['fps']} Hz."
        )
        self.thread = threading.Thread(
            target=self._run, args=(repo_id, replace), daemon=True
        )
        self.thread.start()
        return self.snapshot()

    def start_sync(self, repo_id: str) -> dict:
        repo_id = str(repo_id or "").strip()
        if not self.store.is_hub(repo_id):
            raise RuntimeError("not a Hugging Face dataset")
        hub_repo = str(self.store._read_dataset_meta(repo_id).get("hub_repo") or repo_id)
        return self.start(hub_repo, replace=True)

    def _run(self, repo_id: str, replace: bool) -> None:
        tmp = self.store.root(repo_id).parent / (
            self.store.root(repo_id).name + HUB_TMP_SUFFIX
        )
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            self._set_progress("download", 0, 1, f"Download {repo_id}")
            self.log(f"Download {repo_id}…")
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(repo_id, force_cache_sync=replace)
            self.log(f"Downloaded · {dataset.meta.total_episodes} episodes.")
            self._set_progress(
                "convert", 0, int(dataset.meta.total_episodes), "Convert episodes"
            )
            added = convert_hub_dataset(
                dataset, tmp, self.log, progress=self._set_progress
            )
            final = self.store.root(repo_id)
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            self.store._write_dataset_meta(
                repo_id, {"source": HUB_SOURCE, "hub_repo": repo_id}
            )
            self.store.repo_id = repo_id
            self._set_progress("done", added, added, "Done")
            self.log(f"Ready · {added} episodes.", "ok")
        except Exception as error:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            self.log(str(error), "error")
            with self.lock:
                self.error = str(error)
                self.progress["phase"] = "error"
                self.progress["label"] = str(error)
        finally:
            with self.lock:
                self.running = False
