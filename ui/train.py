from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from collect import ACTION_SEMANTICS, EPISODE_PREFIX, TRAJ_NAME, TRAJECTORY_SCHEMA_VERSION
from core.config import (
    LEROBOT_DIR,
    TRAIN_DIR,
    policy_cameras_from_config,
    policy_rename_map,
    resolve_source,
    resolve_stored,
    stored_path,
)
from devices import resolve_model_device

# ======Settings=========
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = "lerobot/smolvla_base"
DEFAULT_ACT_POLICY = "act"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 8
DEFAULT_LR = 1e-4
DEFAULT_SAVE_EVERY = 1
DEFAULT_RUN = "smolvla"
DEFAULT_ACT_RUN = "act"
DEFAULT_TUNE = "experts"
DEFAULT_NUM_WORKERS = 4
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.95
DEFAULT_OPTIMIZER_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 1e-10
DEFAULT_GRAD_CLIP_NORM = 10.0
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_DECAY_STEPS = 30000
DEFAULT_DECAY_LR = 2.5e-6
DEFAULT_VISION_CACHE = True
DEFAULT_VISION_CACHE_GB = 16
DEFAULT_FRAME_CACHE = True
DEFAULT_FRAME_CACHE_GB = 8
SMOLVLA_BASE_CAMERA_COUNT = 3
SMOLVLA_MAX_CAMERA_COUNT = 5
SMOLVLA_CAMERA_SHAPE = (3, 256, 256)
CAMERA_ORDER = ("front", "wrist")
ROBOT_TYPE = "so100"
ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_DIM = 6
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
LOG_LIMIT = 400
HISTORY_LIMIT = 240
STAT_FRAMES = 8
# ======Settings=========

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_METRIC_RE = re.compile(
    r"(step|smpl|ep|epch|loss|grdn|lr|updt_s|data_s|smp/s|mem_gb):([0-9.+\-eE]+[KMB]?)"
)
_TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_SKIP_LOG = re.compile(r"^(Svt\[|\s*\d+%\|)|Training:\s+\d+%")


def _clean_line(text: str) -> str:
    return _ANSI.sub("", text.replace("\r", "\n")).strip()


def _parse_number(raw: str) -> float:
    text = str(raw or "").strip()
    factor = 1.0
    if text.endswith(("K", "M", "B")):
        factor = {"K": 1e3, "M": 1e6, "B": 1e9}[text[-1]]
        text = text[:-1]
    return float(text) * factor


def _parse_train_metrics(line: str) -> dict | None:
    if "loss:" not in line or "step:" not in line:
        return None
    found = {}
    for key, value in _METRIC_RE.findall(line):
        if key == "step" and value[-1:] in "KMB":
            return None
        found[key] = _parse_number(value)
    if "step" not in found or "loss" not in found:
        return None
    return found


def _parse_tqdm(line: str) -> tuple[int, int] | None:
    if "loss:" in line and "step:" in line:
        return None
    if "step/s" not in line and "Training" not in line:
        return None
    match = _TQDM_RE.search(line)
    if match is None:
        return None
    current = int(match.group(1))
    total = int(match.group(2))
    if total < 1 or current > total:
        return None
    return current, total


def _eta_seconds(current: int, total: int, started: float | None) -> float | None:
    if started is None or current < 1 or total <= current:
        return None
    elapsed = time.monotonic() - started
    if elapsed <= 0:
        return None
    return elapsed / current * (total - current)


def _idle_progress() -> dict:
    return {
        "phase": "idle",
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "etaSeconds": None,
        "label": "",
        "loss": None,
        "lr": None,
        "gradNorm": None,
        "epoch": None,
        "samplesPerS": None,
        "gpuMem": None,
    }


def _slug(value: str, default: str = DEFAULT_RUN) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or default


def _resolve_sft_policy(source: str, mode: str) -> str:
    raw = str(source or "").strip()
    if mode == "act" and raw.lower() in {"", "act", "lerobot/act"}:
        return DEFAULT_ACT_POLICY
    if not raw:
        raise RuntimeError("select a checkpoint")
    try:
        resolved = resolve_source(raw)
    except ValueError:
        resolved = raw
    path = resolved if isinstance(resolved, Path) else Path(raw)
    if path.is_dir():
        nested = path / "pretrained_model"
        if (nested / "model.safetensors").is_file():
            path = nested
        try:
            return stored_path(path)
        except ValueError:
            return str(path.resolve())
    return raw


def _run_has_weights(output: Path) -> bool:
    ckpt = output / "checkpoints"
    if not ckpt.is_dir():
        return False
    return any(
        (item / "pretrained_model" / "model.safetensors").is_file()
        for item in ckpt.iterdir()
        if item.is_dir()
    )


def _lerobot_policy_device(device: str) -> tuple[str, dict[str, str]]:
    raw = str(device or "cpu").strip()
    extra: dict[str, str] = {}
    if raw.startswith("cuda"):
        if ":" in raw:
            extra["CUDA_VISIBLE_DEVICES"] = raw.split(":", 1)[1]
        return "cuda", extra
    return raw, extra


def _amp_supported(device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    import torch

    return bool(torch.cuda.is_available())


def _training_cameras(episodes: list[dict], mapped: list[str] | None = None) -> list[str]:
    if mapped:
        cameras = [name for name in mapped if any(name in item["cameras"] for item in episodes)]
        if cameras:
            return cameras[:SMOLVLA_MAX_CAMERA_COUNT]
    cameras = [name for name in CAMERA_ORDER if any(name in item["cameras"] for item in episodes)]
    seen = set(cameras)
    for item in episodes:
        for name in item["cameras"]:
            if name not in seen:
                cameras.append(name)
                seen.add(name)
    return cameras[:SMOLVLA_MAX_CAMERA_COUNT]


def _first_video(episodes: list[dict], cameras: list[str]) -> Path:
    for name in cameras:
        for item in episodes:
            path = item["folder"] / f"{name}.mp4"
            if path.is_file():
                return path
    raise RuntimeError("no camera videos in selected episodes")


def _write_blank_video(
    path: Path,
    width: int,
    height: int,
    fps: int,
    frames: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode or not path.is_file():
        raise RuntimeError(f"cannot write blank video {path.name}")


def _smolvla_camera_args(cameras: list[str]) -> tuple[list[str], dict[str, str]]:
    if not cameras:
        raise RuntimeError("no cameras selected for training")
    if len(cameras) > SMOLVLA_MAX_CAMERA_COUNT:
        raise RuntimeError(
            f"SmolVLA training supports at most {SMOLVLA_MAX_CAMERA_COUNT} cameras"
        )
    rename_map = policy_rename_map(cameras)
    args = [f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}"]
    extra_features = {
        f"observation.images.camera{index}": {
            "type": "VISUAL",
            "shape": list(SMOLVLA_CAMERA_SHAPE),
        }
        for index in range(SMOLVLA_BASE_CAMERA_COUNT + 1, len(cameras) + 1)
    }
    if extra_features:
        args.append(
            "--policy.input_features="
            + json.dumps(extra_features, separators=(",", ":"))
        )
    return args, rename_map


def _probe_video(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            FFPROBE_BIN,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        raise RuntimeError(f"cannot read video {path.name}")
    width, height = [int(part) for part in line[0].split(",")[:2]]
    return width, height


def _count_frames(path: Path) -> int:
    for extra in (
        ["-show_entries", "stream=nb_frames"],
        ["-count_packets", "-show_entries", "stream=nb_read_packets"],
    ):
        proc = subprocess.run(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                *extra,
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            continue
        value = raw[0].split(",")[0].strip()
        if value.isdigit():
            return int(value)
    raise RuntimeError(f"cannot count frames in {path.name}")


def _sample_jpegs(path: Path, dest: Path, count: int) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "frame_%03d.jpg"
    proc = subprocess.run(
        [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=1",
            "-frames:v",
            str(count),
            str(pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    files = sorted(dest.glob("frame_*.jpg"))
    if not files:
        proc = subprocess.run(
            [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                str(count),
                str(pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        files = sorted(dest.glob("frame_*.jpg"))
    if proc.returncode or not files:
        raise RuntimeError(f"cannot sample {path.name}")
    return [str(item) for item in files]


def _collect_trainable(store, repo_ids: list[str]) -> tuple[list[dict], list[str]]:
    episodes = []
    skipped = []
    for repo_id in repo_ids:
        for item in store.list_episodes(repo_id):
            folder = store.root(repo_id) / f"{EPISODE_PREFIX}{item['index']}"
            traj_path = folder / TRAJ_NAME
            if not traj_path.is_file():
                skipped.append(f"{repo_id} #{item['index']}")
                continue
            meta_path = folder / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            try:
                traj = np.load(traj_path)
                actions = np.asarray(traj["action"])
                states = np.asarray(traj["state"])
            except Exception:
                skipped.append(f"{repo_id} #{item['index']} (unreadable trajectory)")
                continue
            if (
                actions.ndim != 2
                or states.ndim != 2
                or actions.shape[0] != states.shape[0]
                or actions.shape[-1] < ACTION_DIM
                or states.shape[-1] < ACTION_DIM
            ):
                skipped.append(f"{repo_id} #{item['index']} (bad trajectory)")
                continue
            if (
                int(meta.get("trajectory_schema_version") or 0)
                not in (0, TRAJECTORY_SCHEMA_VERSION)
                or (
                    meta.get("action_semantics")
                    and meta.get("action_semantics") != ACTION_SEMANTICS
                )
            ):
                skipped.append(
                    f"{repo_id} #{item['index']} (legacy trajectory format)"
                )
                continue
            cameras = [
                name
                for name in (item.get("cameras") or meta.get("cameras") or [])
                if (folder / f"{name}.mp4").is_file()
            ]
            if not cameras:
                skipped.append(f"{repo_id} #{item['index']}")
                continue
            episodes.append(
                {
                    "repoId": repo_id,
                    "index": int(item["index"]),
                    "id": item.get("id"),
                    "task": str(item.get("task") or meta.get("task") or ""),
                    "fps": int(meta.get("fps") or item.get("fps") or 0),
                    "cameras": cameras,
                    "folder": folder,
                }
            )
    return episodes, skipped


def _act_collect(
    store, repo_ids: list[str], log, mapped_cameras: list[str] | None = None
) -> tuple[list[dict], int, list[str]]:
    episodes, skipped = _collect_trainable(store, repo_ids)
    for item in skipped:
        log(f"Skip {item}: no usable trajectory.", "warn")
    if not episodes:
        raise RuntimeError(
            "no trainable episodes. Record again — older episodes have video only."
        )
    order = list(mapped_cameras or CAMERA_ORDER)
    cameras = [name for name in order if any(name in item["cameras"] for item in episodes)]
    if not cameras:
        raise RuntimeError("no cameras in selected episodes")
    kept = []
    frames = 0
    for item in episodes:
        missing = [name for name in cameras if name not in item["cameras"]]
        if missing:
            log(
                f"Skip {item['repoId']} #{item['index']}: missing {', '.join(missing)}.",
                "warn",
            )
            continue
        traj = np.load(item["folder"] / TRAJ_NAME)
        frames += int(np.asarray(traj["action"]).shape[0])
        kept.append(item)
    if not kept:
        raise RuntimeError("no ACT episodes with required cameras")
    return kept, frames, cameras


def convert_collect_datasets(
    store,
    repo_ids: list[str],
    run: str,
    log,
    progress=None,
    mapped_cameras: list[str] | None = None,
) -> tuple[str, Path, int, int, list[str]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_ids = [str(item).strip() for item in repo_ids if str(item).strip()]
    if not repo_ids:
        raise RuntimeError("select at least one dataset")
    episodes, skipped = _collect_trainable(store, repo_ids)
    for item in skipped:
        log(f"Skip {item}: no usable trajectory.", "warn")
    if not episodes:
        raise RuntimeError(
            "no trainable episodes. Record again — older episodes have video only."
        )
    log(f"Found {len(episodes)} episodes with trajectories.")
    cameras = _training_cameras(episodes, mapped_cameras)
    if not cameras:
        raise RuntimeError("no cameras in selected episodes")
    unused = sorted(
        {
            name
            for item in episodes
            for name in item["cameras"]
            if name not in cameras
        }
    )
    if unused:
        log(
            f"Skip cameras over SmolVLA limit of {SMOLVLA_MAX_CAMERA_COUNT}: "
            + ", ".join(unused) + ".",
            "warn",
        )
    fps_values = {int(item["fps"]) for item in episodes if item["fps"]}
    if len(fps_values) != 1:
        raise RuntimeError(f"mixed fps in selected episodes: {sorted(fps_values)}")
    fps = fps_values.pop()
    first_video = _first_video(episodes, cameras)
    width, height = _probe_video(first_video)
    repo_id = f"local/{run}"
    root = LEROBOT_DIR / run
    if root.exists():
        log(f"Replace existing converted dataset {root}.")
        shutil.rmtree(root)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
    }
    for name in cameras:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    log(
        f"Convert {len(episodes)} episodes · {', '.join(cameras)} · "
        f"{width}×{height} · {fps} Hz → {root}"
    )
    if progress is not None:
        progress("convert", 0, len(episodes), "Create LeRobot dataset")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        image_writer_threads=0,
    )
    pending_videos: dict[str, Path] = {}
    episode_cameras: dict[int, list[str]] = {}

    def _copy_episode_video(video_key: str, episode_index: int) -> Path:
        source = pending_videos[video_key]
        temp_dir = Path(tempfile.mkdtemp(prefix="robosimvid_", dir=str(root)))
        dest = temp_dir / f"{video_key.replace('.', '_')}.mp4"
        shutil.copy2(source, dest)
        return dest

    dataset.writer._encode_temporary_episode_video = _copy_episode_video
    added = 0
    dropped = 0
    try:
        total = len(episodes)
        for offset, item in enumerate(episodes, start=1):
            label = f"{item['repoId']} #{item['index']}"
            if progress is not None:
                progress("convert", offset, total, f"Convert {label}")
            log(
                f"[{offset}/{total}] {label} · {', '.join(item['cameras'])} · "
                f"{item['fps']} Hz"
            )
            traj = np.load(item["folder"] / TRAJ_NAME)
            actions = np.asarray(traj["action"], dtype=np.float32)
            states = np.asarray(traj["state"], dtype=np.float32)
            length = len(actions)
            if len(states) != length:
                dropped += 1
                log(
                    f"Skip {label}: {len(actions)} actions but {len(states)} states.",
                    "warn",
                )
                continue
            video_paths = {}
            sample_paths = {}
            present_keys = []
            broken = False
            for name in cameras:
                key = f"observation.images.{name}"
                video_path = item["folder"] / f"{name}.mp4"
                if not video_path.is_file():
                    video_path = root / "_pad" / "missing.mp4"
                    if not video_path.is_file():
                        _write_blank_video(video_path, width, height, fps, 1)
                    log(f"Pad {label} {name}: missing, stub 1 frame.")
                    sample_dir = root / "_stats" / f"ep{item['index']}_{name}"
                    sample_paths[key] = _sample_jpegs(video_path, sample_dir, 1)
                    video_paths[key] = video_path
                    continue
                frames = _count_frames(video_path)
                if frames != length:
                    dropped += 1
                    log(
                        f"Skip {label}: {frames} {name} frames but "
                        f"{length} trajectory steps.",
                        "warn",
                    )
                    broken = True
                    break
                video_w, video_h = _probe_video(video_path)
                if video_w != width or video_h != height:
                    raise RuntimeError(
                        f"{label} {name} size mismatch "
                        f"{video_w}×{video_h} != {width}×{height}"
                    )
                video_paths[key] = video_path
                present_keys.append(key)
                sample_dir = root / "_stats" / f"ep{item['index']}_{name}"
                sample_paths[key] = _sample_jpegs(
                    video_path, sample_dir, STAT_FRAMES
                )
            if broken or not present_keys or length < 1:
                continue
            episode_index = int(dataset.meta.total_episodes)
            buffer = dataset.writer._create_episode_buffer(episode_index)
            buffer["size"] = length
            buffer["task"] = [item["task"] or "task"] * length
            buffer["action"] = [row[:ACTION_DIM] for row in actions]
            buffer["observation.state"] = [row[:ACTION_DIM] for row in states]
            buffer["frame_index"] = list(range(length))
            buffer["timestamp"] = [index / fps for index in range(length)]
            for key, paths in sample_paths.items():
                buffer[key] = paths
            pending_videos.clear()
            pending_videos.update(video_paths)
            log(f"Copy videos for {label} · {length} frames")
            dataset.save_episode(buffer, parallel_encoding=False)
            episode_cameras[episode_index] = present_keys
            added += 1
            log(f"Added {label} · {length} frames ({added} kept, {dropped} skipped).")
    finally:
        for extra in ("_stats", "_pad"):
            extra_dir = root / extra
            if extra_dir.exists():
                shutil.rmtree(extra_dir, ignore_errors=True)
        log("Finalize LeRobot dataset.")
        dataset.finalize()
        from train_loop.cameras import write_episode_cameras

        write_episode_cameras(root, episode_cameras)
    if dataset.meta.total_episodes < 1:
        raise RuntimeError("conversion produced no episodes")
    log(
        f"LeRobot dataset ready · {dataset.meta.total_episodes} episodes · "
        f"{dataset.meta.total_frames} frames · skipped {dropped}."
    )
    return (
        repo_id,
        root,
        int(dataset.meta.total_episodes),
        int(dataset.meta.total_frames),
        cameras,
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _checkpoint_step(step_dir: Path) -> int | None:
    payload = _read_json(step_dir / "training_state" / "training_step.json")
    if payload.get("step") is not None:
        return int(payload["step"])
    if step_dir.name.isdigit():
        return int(step_dir.name)
    return None


def _epoch_scale(cfg: dict) -> tuple[int, int] | None:
    steps = int(cfg.get("steps") or 0)
    batch = int(cfg.get("batch_size") or 0)
    save_freq = int(cfg.get("save_freq") or 0)
    frames = 0
    root = (cfg.get("dataset") or {}).get("root")
    if root:
        frames = int(
            _read_json(Path(root) / "meta" / "info.json").get("total_frames") or 0
        )
    if frames > 0 and batch > 0:
        steps_per_epoch = math.ceil(frames / batch)
    elif save_freq > 0:
        steps_per_epoch = save_freq
    else:
        return None
    if steps_per_epoch < 1 or steps < 1:
        return None
    return steps_per_epoch, max(1, math.ceil(steps / steps_per_epoch))


def _grpo_update(step_dir: Path) -> int | None:
    iter_path = step_dir / "ITER.txt"
    if iter_path.is_file():
        try:
            return int(iter_path.read_text().strip())
        except ValueError:
            pass
    payload = _read_json(step_dir / "training_state" / "training_step.json")
    if payload.get("kind") == "grpo" or payload.get("update") is not None:
        raw = payload.get("update", payload.get("step"))
        if raw is not None:
            return int(raw)
    cfg = _read_json(step_dir / "pretrained_model" / "train_config.json")
    if cfg.get("kind") == "grpo":
        raw = cfg.get("update")
        if raw is not None:
            return int(raw)
        return _checkpoint_step(step_dir)
    return None


def _checkpoint_label(
    run: str,
    step: int | None,
    scale: tuple[int, int] | None,
    last: bool,
    grpo_update: int | None = None,
) -> str:
    if grpo_update is not None:
        if last:
            return f"{run} · last · update {grpo_update}"
        return f"{run} · update {grpo_update}"
    if scale is not None and step:
        steps_per_epoch, total_epochs = scale
        epoch = min(total_epochs, max(1, math.ceil(step / steps_per_epoch)))
        if last:
            return f"{run} · last · epoch {epoch}/{total_epochs}"
        return f"{run} · epoch {epoch}/{total_epochs}"
    if last:
        return f"{run} · last"
    if step:
        return f"{run} · step {step}"
    return f"{run} · checkpoint"


def _checkpoint_cameras(model_dir: Path) -> list[str]:
    cfg = _read_json(model_dir / "config.json")
    if str(cfg.get("type") or "") == "act":
        return [str(name) for name in (cfg.get("cameras") or []) if name]
    rename = {}
    for step in (_read_json(model_dir / "policy_preprocessor.json").get("steps") or []):
        step_cfg = (step or {}).get("config") or {}
        if step_cfg.get("rename_map"):
            rename = dict(step_cfg["rename_map"])
            break
    if not rename:
        rename = dict(_read_json(model_dir / "train_config.json").get("rename_map") or {})
    pairs = []
    for src, dst in rename.items():
        src_name = str(src).rsplit(".", 1)[-1]
        dst_name = str(dst).rsplit(".", 1)[-1]
        index = 10_000
        if dst_name.startswith("camera"):
            try:
                index = int(dst_name[6:])
            except ValueError:
                pass
        pairs.append((index, src_name))
    pairs.sort()
    return [name for _, name in pairs]


def _checkpoint_type(model_dir: Path) -> str:
    for name in ("config.json", "train_config.json"):
        path = model_dir / name
        if not path.is_file():
            continue
        data = _read_json(path)
        if name == "train_config.json":
            policy = data.get("policy") or {}
            kind = str(policy.get("type") or "")
        else:
            kind = str(data.get("type") or "")
        if kind:
            return "act" if kind == "act" else "smolvla"
    return "smolvla"


_CKPT_TTL_S = 2.0
_ckpt_lock = threading.Lock()
_ckpt_cache: tuple[float, list[dict]] = (0.0, [])


def list_checkpoints() -> list[dict]:
    global _ckpt_cache
    now = time.monotonic()
    with _ckpt_lock:
        stamp, cached = _ckpt_cache
        if cached and now - stamp < _CKPT_TTL_S:
            return list(cached)
    items = _list_checkpoints()
    with _ckpt_lock:
        _ckpt_cache = (time.monotonic(), items)
    return list(items)


def _list_checkpoints() -> list[dict]:
    items = [
        {
            "id": DEFAULT_POLICY,
            "label": DEFAULT_POLICY,
            "kind": "hub",
            "type": "smolvla",
            "cameras": [],
        }
    ]
    if not TRAIN_DIR.is_dir():
        return items
    for run in sorted(TRAIN_DIR.iterdir()):
        ckpt_root = run / "checkpoints"
        if not ckpt_root.is_dir():
            continue
        last = ckpt_root / "last"
        last_resolved = last.resolve() if last.exists() else None
        last_model = last / "pretrained_model"
        last_update = _grpo_update(last) if last.exists() else None
        scale = None
        if last_update is None:
            scale = _epoch_scale(
                _read_json(last_model / "train_config.json")
                if (last_model / "train_config.json").is_file()
                else {}
            )
        if (last_model / "model.safetensors").is_file():
            items.append(
                {
                    "id": stored_path(last_model),
                    "label": _checkpoint_label(
                        run.name,
                        _checkpoint_step(last),
                        scale,
                        True,
                        last_update,
                    ),
                    "kind": "local",
                    "type": _checkpoint_type(last_model),
                    "cameras": _checkpoint_cameras(last_model),
                }
            )
        step_items = []
        for step_dir in ckpt_root.iterdir():
            if step_dir.name == "last":
                continue
            if last_resolved is not None and step_dir.resolve() == last_resolved:
                continue
            model = step_dir / "pretrained_model"
            if not (model / "model.safetensors").is_file():
                continue
            grpo_update = _grpo_update(step_dir)
            if scale is None and grpo_update is None:
                scale = _epoch_scale(_read_json(model / "train_config.json"))
            step = _checkpoint_step(step_dir)
            step_items.append(
                (
                    grpo_update or step or 0,
                    {
                        "id": stored_path(model),
                        "label": _checkpoint_label(
                            run.name, step, scale, False, grpo_update
                        ),
                        "kind": "local",
                        "type": _checkpoint_type(model),
                        "cameras": _checkpoint_cameras(model),
                    },
                )
            )
        step_items.sort(key=lambda item: item[0], reverse=True)
        items.extend(item for _, item in step_items)
    return items


def _finite_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _run_dir(name: str) -> Path:
    slug = _slug(name)
    if slug != str(name or "").strip():
        raise RuntimeError("bad run name")
    path = (TRAIN_DIR / slug).resolve()
    if path.parent != TRAIN_DIR.resolve():
        raise RuntimeError("bad run path")
    return path


def _run_kind(run: Path) -> str:
    metrics = run / "metrics.jsonl"
    if metrics.is_file():
        try:
            first = metrics.read_text(encoding="utf-8").splitlines()[0]
            payload = json.loads(first)
            if "mean_reward" in payload or "update" in payload:
                return "grpo"
            if "loss" in payload:
                return "sft"
        except Exception:
            pass
    ckpt = run / "checkpoints"
    if ckpt.is_dir():
        for item in ckpt.iterdir():
            if item.is_dir() and (item / "ITER.txt").is_file():
                return "grpo"
    if run.name.startswith("grpo"):
        return "grpo"
    return "sft"


def _read_metrics(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _history_from_metrics(kind: str, rows: list[dict]) -> dict[str, list[dict]]:
    if kind == "grpo":
        mapping = (
            ("reward", "mean_reward"),
            ("success", "success_rate"),
            ("loss", "loss"),
            ("clip", "clip_fraction"),
            ("drift", "drift"),
            ("kl", "kl"),
            ("gradNorm", "grad_norm"),
        )
        step_key = "update"
        keys = ("reward", "success", "loss", "clip", "drift", "kl", "gradNorm")
    else:
        mapping = (
            ("loss", "loss"),
            ("lr", "lr"),
            ("gradNorm", "grad_norm"),
        )
        step_key = "step"
        keys = ("loss", "lr", "gradNorm")
    history = {key: [] for key in keys}
    for row in rows:
        step = row.get(step_key, row.get("step"))
        if step is None:
            continue
        step = int(step)
        for dest, src in mapping:
            value = _finite_number(row.get(src))
            if value is None:
                continue
            history[dest].append({"step": step, "value": value})
    return history


def _run_checkpoints(run: Path) -> list[dict]:
    items = []
    ckpt = run / "checkpoints"
    if not ckpt.is_dir():
        return items
    for step_dir in ckpt.iterdir():
        if step_dir.name == "last" or not step_dir.is_dir():
            continue
        model = step_dir / "pretrained_model"
        if not (model / "model.safetensors").is_file():
            continue
        step = _checkpoint_step(step_dir)
        items.append(
            {
                "id": stored_path(model),
                "label": step_dir.name,
                "step": step,
            }
        )
    items.sort(key=lambda item: item.get("step") or 0, reverse=True)
    return items


def _run_config(run: Path) -> dict:
    last = run / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if last.is_file():
        return _read_json(last)
    ckpt = run / "checkpoints"
    if not ckpt.is_dir():
        return {}
    for step_dir in ckpt.iterdir():
        cfg = step_dir / "pretrained_model" / "train_config.json"
        if cfg.is_file():
            return _read_json(cfg)
    return {}


def _last_finite(rows: list[dict], key: str) -> float | None:
    for row in reversed(rows):
        value = _finite_number(row.get(key))
        if value is not None:
            return value
    return None


def list_runs(busy: set[str] | None = None) -> list[dict]:
    busy = busy or set()
    items = []
    if not TRAIN_DIR.is_dir():
        return items
    for run in sorted(TRAIN_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not run.is_dir():
            continue
        kind = _run_kind(run)
        rows = _read_metrics(run)
        history = _history_from_metrics(kind, rows)
        items.append(
            {
                "id": run.name,
                "kind": kind,
                "busy": run.name in busy,
                "mtime": run.stat().st_mtime,
                "checkpoints": len(_run_checkpoints(run)),
                "points": max((len(series) for series in history.values()), default=0),
                "lastLoss": _last_finite(rows, "loss"),
                "lastReward": _last_finite(rows, "mean_reward"),
            }
        )
    return items


def load_run(name: str) -> dict:
    run = _run_dir(name)
    if not run.is_dir():
        raise RuntimeError(f"run not found: {name}")
    kind = _run_kind(run)
    rows = _read_metrics(run)
    history = _history_from_metrics(kind, rows)
    cfg = _run_config(run)
    return {
        "id": run.name,
        "kind": kind,
        "path": str(run),
        "mtime": run.stat().st_mtime,
        "history": history,
        "checkpoints": _run_checkpoints(run),
        "steps": cfg.get("steps"),
        "batch": cfg.get("batch_size"),
        "lastLoss": _last_finite(rows, "loss"),
        "lastReward": _last_finite(rows, "mean_reward"),
        "lastSuccess": _last_finite(rows, "success_rate"),
    }


def delete_run(name: str, busy: set[str] | None = None) -> None:
    if name in (busy or set()):
        raise RuntimeError("run is active")
    path = _run_dir(name)
    if not path.is_dir():
        raise RuntimeError(f"run not found: {name}")
    shutil.rmtree(path)


class TrainRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.running = False
        self.error: str | None = None
        self.logs: list[dict] = []
        self.policy = DEFAULT_POLICY
        self.epochs = DEFAULT_EPOCHS
        self.steps = 0
        self.batch = DEFAULT_BATCH
        self.lr = DEFAULT_LR
        self.save_every = DEFAULT_SAVE_EVERY
        self.run = DEFAULT_RUN
        self.tune = DEFAULT_TUNE
        self.policy_mode = "smolvla"
        self.repo_ids: list[str] = []
        self.output_dir: str | None = None
        self.progress = _idle_progress()
        self._progress_t0 = time.monotonic()
        self.history: dict[str, list[dict]] = {
            "loss": [],
            "lr": [],
            "gradNorm": [],
        }

    def snapshot(self, store=None, extras: bool = True) -> dict:
        with self.lock:
            payload = {
                "running": self.running,
                "policy": self.policy,
                "epochs": self.epochs,
                "steps": self.steps,
                "batch": self.batch,
                "lr": self.lr,
                "saveEvery": self.save_every,
                "run": self.run,
                "tune": self.tune,
                "repoIds": list(self.repo_ids),
                "outputDir": self.output_dir,
                "error": self.error,
                "logs": list(self.logs[-LOG_LIMIT:]),
                "progress": dict(self.progress),
                "history": {
                    key: list(points) for key, points in self.history.items()
                },
            }
        if extras:
            payload["checkpoints"] = list_checkpoints()
            if store is not None:
                payload["datasets"] = store.list_datasets()
        return payload

    def _progress_eta(self, phase: str, current: int, total: int) -> float | None:
        prev_phase = self.progress.get("phase")
        prev_current = int(self.progress.get("current") or 0)
        if phase != prev_phase or current < prev_current:
            self._progress_t0 = time.monotonic()
        eta = _eta_seconds(current, total, self._progress_t0)
        return None if eta is None else round(eta)

    def _set_progress(
        self,
        phase: str,
        current: int,
        total: int,
        label: str,
    ) -> None:
        percent = 0.0
        if total > 0:
            percent = min(100.0, round(100.0 * current / total, 1))
        with self.lock:
            eta = self._progress_eta(phase, current, total)
            self.progress.update(
                {
                    "phase": phase,
                    "current": int(current),
                    "total": int(total),
                    "percent": percent,
                    "etaSeconds": eta,
                    "label": label,
                }
            )

    def _apply_metrics(self, metrics: dict):
        step = int(metrics["step"])
        loss = float(metrics["loss"])
        lr = metrics.get("lr")
        grad = metrics.get("grdn")
        epoch = metrics.get("epch")
        speed = metrics.get("smp/s")
        mem = metrics.get("mem_gb")
        total = self.steps or step
        percent = min(100.0, round(100.0 * step / total, 1)) if total else 0.0
        eta = self._progress_eta("train", step, int(total))
        self.progress.update(
            {
                "phase": "train",
                "current": step,
                "total": int(total),
                "percent": percent,
                "etaSeconds": eta,
                "label": f"Step {step}/{total}",
                "loss": loss,
                "lr": None if lr is None else float(lr),
                "gradNorm": None if grad is None else float(grad),
                "epoch": None if epoch is None else float(epoch),
                "samplesPerS": None if speed is None else float(speed),
                "gpuMem": None if mem is None else float(mem),
            }
        )
        point = {"step": step, "value": loss}
        self.history["loss"].append(point)
        if lr is not None:
            self.history["lr"].append({"step": step, "value": float(lr)})
        if grad is not None:
            self.history["gradNorm"].append({"step": step, "value": float(grad)})
        for key in self.history:
            self.history[key] = self.history[key][-HISTORY_LIMIT:]
        if not self.output_dir:
            return None
        row = {"step": step, "loss": loss}
        if lr is not None:
            row["lr"] = float(lr)
        if grad is not None:
            row["grad_norm"] = float(grad)
        return resolve_stored(self.output_dir) / "metrics.jsonl", row

    def log(self, text: str, kind: str = "info") -> None:
        line = _clean_line(text)
        if not line:
            return
        tqdm_hit = _parse_tqdm(line)
        metrics = _parse_train_metrics(line)
        metric_file = None
        with self.lock:
            if metrics is not None:
                metric_file = self._apply_metrics(metrics)
            elif tqdm_hit is not None:
                current, total = tqdm_hit
                if self.steps:
                    total = self.steps
                self.progress.update(
                    {
                        "phase": "train",
                        "current": current,
                        "total": int(total),
                        "percent": min(100.0, round(100.0 * current / total, 1)),
                        "etaSeconds": self._progress_eta("train", current, int(total)),
                        "label": f"Step {current}/{total}",
                    }
                )
                return
            if _SKIP_LOG.search(line):
                pass
            else:
                self.logs.append({"kind": kind, "text": line})
                self.logs = self.logs[-LOG_LIMIT:]
        if metric_file is not None:
            path, row = metric_file
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def start(self, payload: dict, store, compute: dict, training: dict) -> dict:
        with self.lock:
            if self.running:
                raise RuntimeError("training is already running")
            self.policy_mode = (
                "act"
                if str(payload.get("policy_mode") or "").strip().lower() == "act"
                else "smolvla"
            )
            default_policy = DEFAULT_ACT_POLICY if self.policy_mode == "act" else DEFAULT_POLICY
            default_run = DEFAULT_ACT_RUN if self.policy_mode == "act" else DEFAULT_RUN
            self.policy = _resolve_sft_policy(
                str(payload.get("policy") or default_policy).strip(),
                self.policy_mode,
            )
            requested_run = str(payload.get("run") or "").strip()
            self.run = _slug(requested_run, default_run)
            self.epochs = max(1, int(payload.get("epochs") or DEFAULT_EPOCHS))
            self.steps = 0
            self.batch = max(1, int(payload.get("batch") or DEFAULT_BATCH))
            self.lr = max(1e-8, float(payload.get("lr") or DEFAULT_LR))
            self.save_every = max(0, int(payload.get("save_every") or payload.get("saveEvery") or 0))
            if "save_every" not in payload and "saveEvery" not in payload:
                self.save_every = DEFAULT_SAVE_EVERY
            tune = str(payload.get("tune") or DEFAULT_TUNE).strip().lower()
            self.tune = "full" if tune == "full" else "experts"
            self.repo_ids = [
                str(item).strip()
                for item in (payload.get("repo_ids") or payload.get("repoIds") or [])
                if str(item).strip()
            ]
            self.error = None
            self.output_dir = None
            self.logs = []
            self.progress = _idle_progress()
            self._progress_t0 = time.monotonic()
            self.history = {"loss": [], "lr": [], "gradNorm": []}
            self.running = True
        self._set_progress("prepare", 0, 1, f"Prepare {self.run}")
        self.log(f"Prepare {self.run}.")
        self.thread = threading.Thread(
            target=self._run,
            args=(store, compute, dict(training)),
            daemon=True,
        )
        self.thread.start()
        return self.snapshot(store)

    def stop(self) -> dict:
        with self.lock:
            proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.log("Stopped.", "warn")
        with self.lock:
            self.running = False
            self.proc = None
        return self.snapshot()

    def _pump(self, stream) -> None:
        buf = ""
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            buf += chunk.replace("\r", "\n")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self.log(line)
        if buf.strip():
            self.log(buf)

    def _run(self, store, compute: dict, training: dict) -> None:
        try:
            output = TRAIN_DIR / self.run
            if output.exists():
                if self.policy_mode == "act" and not _run_has_weights(output):
                    self.log(f"Replace unfinished run {output}.", "warn")
                    shutil.rmtree(output)
                else:
                    raise RuntimeError(f"run exists: {output}")
            if self.policy_mode == "act":
                import torch

                from model.act import run_act_sft

                episodes, frames, cameras = _act_collect(
                    store,
                    self.repo_ids,
                    self.log,
                    policy_cameras_from_config({"compute": compute}),
                )
                with self.lock:
                    still_running = self.running
                if not still_running:
                    self.log("Stopped before training started.", "warn")
                    return
                output.parent.mkdir(parents=True, exist_ok=True)
                device = resolve_model_device(compute.get("model"))
                policy_device, device_env = _lerobot_policy_device(device)
                if device_env:
                    os.environ.update(device_env)
                steps_per_epoch = max(1, math.ceil(max(1, frames) / self.batch))
                steps = max(1, self.epochs * steps_per_epoch)
                with self.lock:
                    self.steps = steps
                    self.output_dir = stored_path(output)
                self._set_progress("prepare", 0, len(episodes), "Load collect episodes")
                self.log(
                    f"ACT SFT · {len(episodes)} episodes · {frames} frames · "
                    f"no LeRobot convert · batch {self.batch} → {steps} steps."
                )
                if self.policy.lower() in {"act", "lerobot/act"}:
                    self.log("Init from scratch.")
                else:
                    self.log(f"Init from {self.policy}.")

                def should_stop() -> bool:
                    with self.lock:
                        return not self.running

                run_act_sft(
                    episodes,
                    output,
                    torch.device(policy_device),
                    self.epochs,
                    self.batch,
                    self.lr,
                    self.save_every,
                    self.policy,
                    cameras,
                    self.log,
                    self._set_progress,
                    should_stop,
                    training=training,
                )
                if should_stop():
                    self.log("Stopped.", "warn")
                    return
                self._set_progress("done", steps, steps, "Done")
                self.log(f"Done. Checkpoints in {output}", "ok")
                return
            repo_id, root, count, frames, cameras = convert_collect_datasets(
                store,
                self.repo_ids,
                self.run,
                self.log,
                progress=self._set_progress,
                mapped_cameras=policy_cameras_from_config({"compute": compute}),
            )
            with self.lock:
                still_running = self.running
            if not still_running:
                self.log("Stopped before training started.", "warn")
                return
            output.parent.mkdir(parents=True, exist_ok=True)
            device = resolve_model_device(compute.get("model"))
            policy_device, device_env = _lerobot_policy_device(device)
            camera_args, rename_map = _smolvla_camera_args(cameras)
            use_amp = bool(training.get("use_amp", True))
            if use_amp and not _amp_supported(policy_device):
                use_amp = False
                self.log(
                    "AMP off: this GPU has no BF16. LeRobot AMP would crash on V100.",
                    "warn",
                )
            steps_per_epoch = max(1, math.ceil(max(1, frames) / self.batch))
            steps = max(1, self.epochs * steps_per_epoch)
            num_workers = max(0, int(training.get("num_workers", DEFAULT_NUM_WORKERS)))
            vision_cache = bool(training.get("vision_cache", DEFAULT_VISION_CACHE))
            vision_cache_gb = max(
                0.0, float(training.get("vision_cache_gb", DEFAULT_VISION_CACHE_GB))
            )
            if vision_cache_gb <= 0:
                vision_cache = False
            frame_cache = bool(training.get("frame_cache", DEFAULT_FRAME_CACHE))
            frame_cache_gb = max(
                0.0, float(training.get("frame_cache_gb", DEFAULT_FRAME_CACHE_GB))
            )
            if frame_cache_gb <= 0:
                frame_cache = False
            beta1 = float(training.get("optimizer_beta1", DEFAULT_BETA1))
            beta2 = float(training.get("optimizer_beta2", DEFAULT_BETA2))
            optimizer_eps = max(
                0.0, float(training.get("optimizer_eps", DEFAULT_OPTIMIZER_EPS))
            )
            weight_decay = max(
                0.0,
                float(training.get("optimizer_weight_decay", DEFAULT_WEIGHT_DECAY)),
            )
            grad_clip_norm = max(
                0.0, float(training.get("grad_clip_norm", DEFAULT_GRAD_CLIP_NORM))
            )
            warmup = max(
                0, int(training.get("scheduler_warmup_steps", DEFAULT_WARMUP_STEPS))
            )
            warmup = min(warmup, max(0, steps - 1))
            decay_steps = max(1, steps)
            decay_lr = max(
                0.0, float(training.get("scheduler_decay_lr", DEFAULT_DECAY_LR))
            )
            save_freq = 0
            if self.save_every > 0:
                save_freq = steps_per_epoch * self.save_every
            with self.lock:
                self.steps = steps
            self._set_progress("train", 0, steps, f"Start training · {steps} steps")
            command = [
                sys.executable,
                "-m",
                "train_loop.lerobot_train",
                f"--policy.path={self.policy}",
                f"--dataset.repo_id={repo_id}",
                f"--dataset.root={stored_path(root)}",
                f"--output_dir={stored_path(output)}",
                f"--batch_size={self.batch}",
                f"--steps={steps}",
                f"--job_name={self.run}",
                f"--policy.device={policy_device}",
                "--policy.push_to_hub=false",
                "--wandb.enable=false",
                *camera_args,
                f"--save_freq={save_freq}",
                f"--num_workers={num_workers}",
                f"--log_freq={max(1, min(10, steps))}",
                f"--policy.train_expert_only={'true' if self.tune == 'experts' else 'false'}",
                f"--policy.freeze_vision_encoder={'true' if self.tune == 'experts' else 'false'}",
                f"--policy.use_amp={'true' if use_amp else 'false'}",
                f"--policy.optimizer_lr={self.lr}",
                f"--policy.optimizer_betas=[{beta1},{beta2}]",
                f"--policy.optimizer_eps={optimizer_eps}",
                f"--policy.optimizer_weight_decay={weight_decay}",
                f"--policy.optimizer_grad_clip_norm={grad_clip_norm}",
                f"--policy.scheduler_warmup_steps={warmup}",
                f"--policy.scheduler_decay_steps={decay_steps}",
                f"--policy.scheduler_decay_lr={decay_lr}",
            ]
            scope = "experts only" if self.tune == "experts" else "full model"
            self.log(f"Init from {self.policy}.")
            self.log(
                f"{self.epochs} epochs · {frames} frames · batch {self.batch} → {steps} steps."
            )
            self.log(
                f"Train {count} episodes on {device} · {scope} · lr {self.lr:g} · "
                f"warmup {warmup} · decay {decay_steps} · "
                f"save every {save_freq or 'last only'} steps · "
                f"vision cache {'off' if not vision_cache else f'{vision_cache_gb:g} GB'} · "
                f"frame cache {'off' if not frame_cache else f'{frame_cache_gb:g} GB total'}."
            )
            self.log(
                "Camera map · "
                + ", ".join(
                    f"{source.rsplit('.', 1)[-1]} → {target.rsplit('.', 1)[-1]}"
                    for source, target in rename_map.items()
                )
            )
            self.log("Launch train_loop.")
            with self.lock:
                self.output_dir = stored_path(output)
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    **device_env,
                    "ROBOSIM_VISION_CACHE": "1" if vision_cache else "0",
                    "ROBOSIM_VISION_CACHE_GB": str(vision_cache_gb),
                    "ROBOSIM_FRAME_CACHE": "1" if frame_cache else "0",
                    "ROBOSIM_FRAME_CACHE_GB": str(frame_cache_gb),
                },
            )
            with self.lock:
                self.proc = proc
            self._pump(proc.stdout)
            code = proc.wait()
            if code:
                raise RuntimeError(f"lerobot-train exited {code}")
            self._set_progress("done", steps, steps, "Done")
            self.log(f"Done. Checkpoints in {output}", "ok")
        except Exception as error:
            self.log(str(error), "error")
            with self.lock:
                self.error = str(error)
                self.progress["phase"] = "error"
                self.progress["label"] = str(error)
        finally:
            with self.lock:
                self.running = False
                self.proc = None
