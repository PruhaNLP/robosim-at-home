#!/usr/bin/env python3
import argparse
import io
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

# ======Settings=========
REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = REPO_ROOT / "sim"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
COLLISION_GROUP = 3
VIEW_SIZE = 512
VIEW_JPEG_QUALITY = 85
STREAM_BOUNDARY = "robosimframe"
STREAM_MAX_FPS = 30
STATS_WINDOW = 1.0
ROBOT_ACTION_DIM = 6
WRIST_FLEX_INDEX = 3
GRIPPER_INDEX = 5
LEADER_GRIPPER_MIN_DEG = 0.0
LEADER_GRIPPER_MAX_DEG = 100.0
DEFAULT_COLLECT_FPS = 15
DEFAULT_REPO_ID = "local/so100_collect"
WORKER_REQUEST_S = 30
WORKER_WRITE_S = 15
WORKER_STATUS_S = 3
WORKER_GENERATE_S = 120
WORKER_STOP_S = 45
OBS_RENDER_S = 15
WORKER_PRIORITY_OPS = frozenset(
    {
        "generate",
        "reset",
        "test_stop",
        "test_start",
        "start_record",
        "stop_record",
    }
)
WORKER_DROP_WHEN_PRIORITY = frozenset({"preview"})
IDLE_TEST = {
    "running": False,
    "phase": "idle",
    "seconds": 0.0,
    "success": None,
    "error": None,
    "logs": [],
}
PREVIEW_HZ = 30
RECORD_QUEUE_MAX = 90
RECORD_FRAME_QUEUE = 32
FFMPEG_BIN = "ffmpeg"
FFMPEG_PRESET = "ultrafast"
FFMPEG_CRF = 23
FFMPEG_JOIN_S = 12
FFMPEG_WAIT_S = 8
OBJECT_PREVIEW_PX = 64
# ======Settings=========

_OBJECT_PREVIEW_CACHE: dict[str, bytes] = {}


def payload_cameras(payload) -> list[str]:
    raw = payload.get("cameras") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common import resolve_path
from core.config import (
    COLLECT_SCENE_DIR,
    DATASETS_DIR,
    EVAL_VALSET_DIR,
    LLM_LOG_PATH,
    UI_SCENE_DIR,
    ensure_data_dirs,
    load_config,
    normalize_policy_mode,
    resolve_stored,
    save_config,
    stored_path,
)
from scene.objects import object_index
from devices import (
    apply_render_device,
    cached_model_devices,
    cached_render_devices,
    list_model_devices,
    list_render_devices,
    resolve_model_device,
    resolve_render_device,
)


def object_roots(config: dict) -> dict[str, Path]:
    return {
        "targets": resolve_path(SIM_ROOT, config["paths"]["targets_dir"]),
        "distractors": resolve_path(SIM_ROOT, config["paths"]["distractors_dir"]),
    }


def object_catalog(config: dict) -> dict:
    roots = object_roots(config)
    index = object_index(roots["targets"], roots["distractors"])
    return {"items": [{"id": name} for name in sorted(index)]}


def object_preview(name: str, config: dict) -> bytes:
    cached = _OBJECT_PREVIEW_CACHE.get(name)
    if cached is not None:
        return cached
    if not name or Path(name).name != name:
        raise ValueError("invalid object id")
    roots = object_roots(config)
    index = object_index(roots["targets"], roots["distractors"])
    object_dir = index.get(name)
    if object_dir is None:
        raise FileNotFoundError(name)
    texture = object_dir / "texture.png"
    if not texture.is_file():
        raise FileNotFoundError(name)
    with Image.open(texture) as image:
        image = image.convert("RGBA")
        image.thumbnail((OBJECT_PREVIEW_PX, OBJECT_PREVIEW_PX))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
    _OBJECT_PREVIEW_CACHE[name] = payload
    return payload


class RenderWorker:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.device_id: str | None = None
        self.boot = threading.Lock()
        self.write_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.pending: dict[int, queue.Queue] = {}
        self.next_id = 0
        self.reader: threading.Thread | None = None

    def ensure(self, compute: dict) -> None:
        device = resolve_render_device(compute)
        with self.boot:
            if (
                self.proc is not None
                and self.proc.poll() is None
                and self.device_id == device["id"]
            ):
                return
            self.close()
            env = os.environ.copy()
            apply_render_device(device, env)
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=env,
                cwd=str(REPO_ROOT),
            )
            self.device_id = device["id"]
            self.reader = threading.Thread(target=self._read_loop, daemon=True)
            self.reader.start()

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        self.device_id = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=4)
            except Exception:
                proc.kill()
        with self.pending_lock:
            waiting = list(self.pending.values())
            self.pending.clear()
        for item in waiting:
            item.put({"ok": False, "error": "render worker closed", "blobs": []})
        if TEST.snapshot().get("running"):
            TEST.apply(
                {
                    "running": False,
                    "phase": "error",
                    "error": "render worker closed",
                }
            )
            set_worker_mode("idle")
        else:
            TEST.reset_idle()
        PREVIEW.clear_sim_slots()

    def request(self, payload: dict, timeout: float = WORKER_REQUEST_S) -> dict:
        if (
            self.proc is None
            or self.proc.poll() is not None
            or self.proc.stdin is None
            or self.proc.stdout is None
        ):
            raise RuntimeError("render worker is not running")
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.next_id += 1
            req_id = self.next_id
            self.pending[req_id] = waiter
        message = dict(payload)
        message["id"] = req_id
        write_s = min(WORKER_WRITE_S, max(1.0, float(timeout)))
        acquired = self.write_lock.acquire(timeout=write_s)
        if not acquired:
            with self.pending_lock:
                self.pending.pop(req_id, None)
            raise TimeoutError("render worker is busy")
        try:
            if (
                self.proc is None
                or self.proc.poll() is not None
                or self.proc.stdin is None
            ):
                raise RuntimeError("render worker is not running")
            _, writable, _ = select.select(
                [], [self.proc.stdin.fileno()], [], write_s
            )
            if not writable:
                raise TimeoutError("render worker is busy")
            self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except Exception:
            with self.pending_lock:
                self.pending.pop(req_id, None)
            raise
        finally:
            self.write_lock.release()
        try:
            reply = waiter.get(timeout=timeout)
        except queue.Empty as error:
            with self.pending_lock:
                self.pending.pop(req_id, None)
            raise TimeoutError(
                f"render worker timed out after {timeout:.0f}s"
            ) from error
        except Exception:
            with self.pending_lock:
                self.pending.pop(req_id, None)
            raise
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "render worker failed")
        return reply

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                header = json.loads(line.decode("utf-8"))
                blobs = []
                for size in header.get("blobs") or []:
                    blobs.append(_readexact(proc.stdout, int(size)))
                header["blobs"] = blobs
                if header.get("event") == "preview":
                    PREVIEW.put(
                        str(header.get("slot") or "infer"),
                        list(header.get("cameras") or []),
                        blobs,
                    )
                    continue
                if header.get("event") == "test_status":
                    TEST.apply(header)
                    continue
                req_id = header.get("id")
                waiter = None
                if req_id is not None:
                    with self.pending_lock:
                        waiter = self.pending.pop(int(req_id), None)
                if waiter is not None:
                    waiter.put(header)
        except Exception:
            pass
        with self.pending_lock:
            waiting = list(self.pending.values())
            self.pending.clear()
        for item in waiting:
            item.put({"ok": False, "error": "render worker exited", "blobs": []})
        if TEST.snapshot().get("running"):
            TEST.apply(
                {
                    "running": False,
                    "phase": "error",
                    "error": "render worker exited",
                }
            )
            set_worker_mode("idle")


class ViewStats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.total = 0
        self.per_total: dict[str, int] = {}
        self.stamps: dict[str, list[float]] = {}

    def reset(self) -> None:
        with self.lock:
            self.total = 0
            self.per_total = {}
            self.stamps = {}

    def add(self, camera: str) -> None:
        now = time.monotonic()
        with self.lock:
            self.total += 1
            self.per_total[camera] = self.per_total.get(camera, 0) + 1
            recent = [stamp for stamp in self.stamps.get(camera, []) if now - stamp <= STATS_WINDOW]
            recent.append(now)
            self.stamps[camera] = recent

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            hz = 0.0
            for camera, stamps in list(self.stamps.items()):
                recent = [stamp for stamp in stamps if now - stamp <= STATS_WINDOW]
                self.stamps[camera] = recent
                rate = len(recent) / STATS_WINDOW
                if rate > hz:
                    hz = rate
            return {"frames": self.total, "hz": round(hz, 1)}


class PreviewCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: dict[str, dict[str, bytes]] = {
            "collect": {},
            "infer": {},
            "grpo": {},
        }
        self.scenes: dict[str, dict | None] = {
            "collect": None,
            "infer": None,
            "grpo": None,
        }
        self.gens: dict[str, int] = {
            "collect": 0,
            "infer": 0,
            "grpo": 0,
        }
        self.frame_gens: dict[str, int] = {
            "collect": 0,
            "infer": 0,
            "grpo": 0,
        }

    def put(self, slot: str, cameras: list[str], blobs: list[bytes]) -> None:
        if not cameras or len(cameras) != len(blobs):
            return
        with self.lock:
            current = dict(self.frames.get(slot) or {})
            for name, blob in zip(cameras, blobs):
                if blob:
                    current[name] = blob
            self.frames[slot] = current
            self.frame_gens[slot] = int(self.frame_gens.get(slot, 0)) + 1

    def get(self, slot: str, camera: str) -> bytes | None:
        with self.lock:
            return self.frames.get(slot, {}).get(camera)

    def set_scene(self, slot: str, scene: dict | None) -> None:
        def identity(value):
            if not isinstance(value, dict):
                return None
            return (
                value.get("seed"),
                value.get("instruction"),
                tuple(value.get("cameras") or []),
            )

        with self.lock:
            if identity(self.scenes.get(slot)) != identity(scene):
                self.gens[slot] = int(self.gens.get(slot, 0)) + 1
            self.scenes[slot] = scene

    def scene(self, slot: str) -> dict | None:
        with self.lock:
            return self.scenes.get(slot)

    def scene_gen(self, slot: str) -> int:
        with self.lock:
            return int(self.gens.get(slot, 0))

    def frame_gen(self, slot: str) -> int:
        with self.lock:
            return int(self.frame_gens.get(slot, 0))

    def clear_slot(self, slot: str) -> None:
        with self.lock:
            self.frames[slot] = {}
            self.scenes[slot] = None
            self.gens[slot] = int(self.gens.get(slot, 0)) + 1
            self.frame_gens[slot] = int(self.frame_gens.get(slot, 0)) + 1

    def clear_sim_slots(self) -> None:
        self.clear_slot("collect")
        self.clear_slot("infer")


class TestRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = dict(IDLE_TEST)

    def snapshot(self) -> dict:
        with self.lock:
            payload = dict(self.status)
            payload["logs"] = list(self.status.get("logs") or [])
            return payload

    def apply(self, payload: dict | None) -> dict:
        if payload:
            with self.lock:
                for key in ("running", "phase", "seconds", "success", "error"):
                    if key in payload:
                        self.status[key] = payload[key]
                if "logs" in payload:
                    incoming = list(payload.get("logs") or [])
                    current = self.status.get("logs") or []
                    if incoming or not current:
                        self.status["logs"] = incoming
        snap = self.snapshot()
        if COLLECT is not None:
            if snap.get("running"):
                COLLECT.quiet.set()
            else:
                COLLECT.quiet.clear()
        return snap

    def reset_idle(self) -> dict:
        with self.lock:
            self.status = {**IDLE_TEST, "logs": []}
        if COLLECT is not None:
            COLLECT.quiet.clear()
        return self.snapshot()


def _readexact(stream, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            raise RuntimeError("render worker exited")
        buf.extend(chunk)
    return bytes(buf)


class CollectRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.preview_thread: threading.Thread | None = None
        self.recording = False
        self.episode = 0
        self.frames = 0
        self.fps = DEFAULT_COLLECT_FPS
        self.task = ""
        self.repo_id = DEFAULT_REPO_ID
        self.error: str | None = None
        self.logs: list[dict] = []
        self.traj_action: list[list[float]] = []
        self.traj_state: list[list[float]] = []
        self.stopping_record = False
        self.quiet = threading.Event()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.thread is not None and self.thread.is_alive(),
                "recording": self.recording,
                "stopping": self.stopping_record,
                "paused": self.quiet.is_set(),
                "episode": self.episode,
                "frames": self.frames,
                "fps": self.fps,
                "task": self.task,
                "repoId": self.repo_id,
                "error": self.error,
                "logs": list(self.logs[-80:]),
            }

    def log(self, text: str, kind: str = "info") -> None:
        with self.lock:
            self.logs.append({"kind": kind, "text": text})
            self.logs = self.logs[-80:]

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop.clear()
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.thread.start()
        self.preview_thread.start()

    def stop_join(self) -> None:
        self.stop.set()
        for thread in (self.thread, self.preview_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=8)
        self.thread = None
        self.preview_thread = None
        with self.lock:
            self.recording = False

    def _control_loop(self) -> None:
        while not self.stop.is_set():
            with self.lock:
                interval = 1.0 / max(1, int(self.fps))
                hz = int(self.fps)
                recording = self.recording
                pausing = self.stopping_record
            tick = time.perf_counter()
            if not pausing and not self.quiet.is_set():
                try:
                    leader_action = LEADER.action_vector()
                    reply = WORKER.request(
                        {
                            "op": "apply_ctrl",
                            "slot": "collect",
                            "action": leader_action,
                            "hz": hz,
                            "record": recording,
                        }
                    )
                    store_preview("collect", reply)
                    if recording and reply.get("recorded"):
                        action = reply.get("action")
                        state = reply.get("state")
                        if action is None or state is None:
                            raise RuntimeError("recorder returned an incomplete sample")
                        with self.lock:
                            self.frames += 1
                            self.traj_action.append(
                                [float(value) for value in action]
                            )
                            self.traj_state.append(
                                [float(value) for value in state]
                            )
                            frames = self.frames
                        if frames % max(1, hz * 5) == 0:
                            self.log(f"Recording · {frames} trajectory steps.")
                except Exception as error:
                    with self.lock:
                        self.error = str(error)
                    self.log(str(error), "error")
                    time.sleep(0.2)
            remain = interval - (time.perf_counter() - tick)
            if remain > 0:
                time.sleep(remain)

    def _preview_loop(self) -> None:
        interval = 1.0 / PREVIEW_HZ
        while not self.stop.is_set():
            tick = time.perf_counter()
            if not self.stopping_record and not self.quiet.is_set():
                try:
                    store_preview(
                        "collect",
                        WORKER.request({"op": "preview", "slot": "collect"}),
                    )
                except Exception:
                    pass
            remain = interval - (time.perf_counter() - tick)
            if remain > 0:
                time.sleep(remain)

    def begin_record(self, payload: dict, cameras: list[str], task: str) -> dict:
        if GRPO is not None and GRPO.running:
            raise RuntimeError("GRPO is running")
        if eval_busy():
            raise RuntimeError("eval is running")
        if TEST.snapshot().get("running"):
            raise RuntimeError("inference test is running")
        with self.lock:
            if self.recording:
                raise RuntimeError("already recording")
            self.repo_id = str(payload.get("repo_id") or DEFAULT_REPO_ID)
            self.fps = max(
                1,
                min(STREAM_MAX_FPS, int(payload.get("fps") or DEFAULT_COLLECT_FPS)),
            )
            self.task = str(payload.get("task") or task or "")
            self.frames = 0
            self.error = None
            self.traj_action = []
            self.traj_state = []
        folder = DATASET.begin_episode(self.repo_id, self.task, self.fps, cameras)
        try:
            WORKER.request(
                {
                    "op": "start_record",
                    "slot": "collect",
                    "folder": stored_path(folder),
                    "cameras": cameras,
                    "fps": self.fps,
                    "width": VIEW_SIZE,
                    "height": VIEW_SIZE,
                }
            )
        except Exception:
            DATASET.discard_episode()
            raise
        with self.lock:
            self.recording = True
            self.episode = DATASET.current_index()
        self.start()
        self.log(
            f"Recording episode {self.episode} · {self.repo_id} · "
            f"{self.fps} Hz · cameras: {', '.join(cameras)}."
        )
        return self.snapshot()

    def apply_settings(self, payload: dict) -> dict:
        with self.lock:
            if self.recording and any(
                key in payload for key in ("fps", "repo_id", "task")
            ):
                raise RuntimeError("recording settings are immutable until the episode stops")
            if payload.get("fps") is not None:
                self.fps = max(1, min(STREAM_MAX_FPS, int(payload["fps"])))
            if payload.get("repo_id") is not None:
                self.repo_id = str(payload["repo_id"] or DEFAULT_REPO_ID)
                DATASET.repo_id = self.repo_id
            if payload.get("task") is not None:
                self.task = str(payload["task"] or "")
        return self.snapshot()

    def end_record(self, save: bool) -> dict:
        with self.lock:
            recording = self.recording
            self.recording = False
            self.stopping_record = True
        if not recording:
            with self.lock:
                self.stopping_record = False
            return self.snapshot()
        reply: dict = {}
        try:
            reply = WORKER.request(
                {"op": "stop_record", "slot": "collect", "save": save},
                timeout=WORKER_STOP_S,
            )
        except Exception as error:
            self.log(str(error), "error")
            DATASET.discard_episode()
            with self.lock:
                self.stopping_record = False
                self.frames = 0
                self.traj_action = []
                self.traj_state = []
                if save:
                    self.error = str(error)
            if save:
                raise
            return self.snapshot()
        try:
            if save:
                with self.lock:
                    actions = list(self.traj_action)
                    states = list(self.traj_state)
                    self.traj_action = []
                    self.traj_state = []
                if actions:
                    DATASET.save_trajectory(actions, states)
                else:
                    self.log(
                        "Episode has no trajectory and cannot be used for training.",
                        "warn",
                    )
                index = DATASET.current_index()
                video_frames = int(reply.get("frames") or 0)
                if video_frames != len(actions):
                    DATASET.discard_episode()
                    raise RuntimeError(
                        "discarded episode: video and trajectory lengths differ "
                        f"({video_frames} != {len(actions)})"
                    )
                DATASET.finish_episode(video_frames)
                with self.lock:
                    self.episode = index
                    self.frames = 0
                details = (
                    f"Saved episode {self.episode} · {reply.get('frames', 0)} video frames"
                    f" · {len(actions)} trajectory steps."
                )
                dropped = int(reply.get("dropped") or 0)
                if dropped:
                    details += f" Dropped {dropped} video frames."
                self.log(details, "warn" if dropped else "ok")
                if reply.get("forced"):
                    self.log(
                        "Video encoder did not stop cleanly; inspect the episode before training.",
                        "warn",
                    )
            else:
                DATASET.discard_episode()
                with self.lock:
                    self.traj_action = []
                    self.traj_state = []
                    self.frames = 0
                self.log("Discarded episode.")
        finally:
            with self.lock:
                self.stopping_record = False
        return self.snapshot()


WORKER = RenderWorker()
WORKER_MODE = "idle"
VIEW_STATS = ViewStats()
PREVIEW = PreviewCache()
TEST = TestRuntime()


def store_preview(slot: str, reply: dict) -> None:
    store_scene(slot, reply.get("scene"))
    PREVIEW.put(slot, list(reply.get("cameras") or []), list(reply.get("blobs") or []))


def store_scene(slot: str, scene: dict | None) -> None:
    if scene is not None:
        PREVIEW.set_scene(slot, scene)


@contextmanager
def pause_collect():
    COLLECT.quiet.set()
    try:
        yield
    finally:
        if not TEST.snapshot().get("running"):
            COLLECT.quiet.clear()


if "--worker" not in sys.argv:
    from collect import DatasetStore, HubImportRuntime
    from leader import LeaderSession
    from grpo_runtime import GrpoRuntime
    from eval_runtime import EvalRuntime
    from train import TrainRuntime, delete_run, list_checkpoints, list_runs, load_run

    LEADER = LeaderSession(COLLECT_SCENE_DIR)
    DATASET = DatasetStore(DATASETS_DIR)
    COLLECT = CollectRuntime()
    COLLECT.repo_id = DATASET.repo_id
    TRAIN = TrainRuntime()
    GRPO = GrpoRuntime()
    EVAL = EvalRuntime()
    HUB_IMPORT = HubImportRuntime(DATASET)

    def _busy_runs() -> set[str]:
        names = set()
        if TRAIN.running:
            names.add(TRAIN.run)
        if GRPO.running:
            names.add(GRPO.run)
        return names
else:
    LEADER = None
    DATASET = None
    COLLECT = None
    TRAIN = None
    GRPO = None
    EVAL = None
    HUB_IMPORT = None
    list_checkpoints = None
    list_runs = None
    load_run = None
    delete_run = None


def public_devices(*, probe: bool = True) -> list[dict]:
    extra = list_render_devices() if probe else cached_render_devices()
    if extra is None:
        extra = []
    return [
        {
            "id": "auto",
            "kind": "auto",
            "label": "Auto",
            "detail": "First working GPU, otherwise CPU",
        },
        *[
            {
                "id": device["id"],
                "kind": device["kind"],
                "label": device["label"],
                "detail": device["detail"],
            }
            for device in extra
        ],
    ]


def public_model_devices(*, probe: bool = True) -> list[dict]:
    extra = list_model_devices() if probe else cached_model_devices()
    if extra is None:
        extra = []
    return [
        {
            "id": "auto",
            "kind": "auto",
            "label": "Auto",
            "detail": "First CUDA GPU, otherwise CPU",
        },
        {
            "id": "cpu",
            "kind": "cpu",
            "label": "CPU",
            "detail": "SmolVLA on CPU",
        },
        *[
            {
                "id": device["id"],
                "kind": device["kind"],
                "label": device["label"],
                "detail": device["detail"],
            }
            for device in extra
        ],
    ]


def patch_config(path, value) -> dict:
    if not isinstance(path, (list, tuple)) or not path:
        raise ValueError("config patch requires a path")
    keys = [str(item) for item in path]
    config = load_config()
    if keys[0] not in config:
        raise ValueError(f"unknown config key {keys[0]}")
    cursor = config
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            raise ValueError(f"bad config path {keys}")
        cursor = next_value
    last = keys[-1]
    if last == "seed" and len(keys) == 1:
        cursor[last] = int(value)
    elif last == "policy_mode" and len(keys) == 1:
        cursor[last] = normalize_policy_mode(value)
    else:
        cursor[last] = value
    save_config(config)
    return load_config()


def public_model_device_id(value: str | None) -> str:
    raw = str(value or "auto")
    if raw in {"auto", "cpu"}:
        return raw
    ids = {device["id"] for device in public_model_devices()}
    if raw in ids:
        return raw
    if raw.lower() == "cuda":
        gpus = list_model_devices()
        return gpus[0]["id"] if gpus else "cpu"
    return "auto"


def slot_output_dir(slot: str) -> Path:
    if slot == "collect":
        return COLLECT_SCENE_DIR
    if slot == "infer":
        return UI_SCENE_DIR
    raise RuntimeError(f"unknown slot {slot}")


def worker_running() -> bool:
    return WORKER.proc is not None and WORKER.proc.poll() is None


def set_worker_mode(mode: str) -> None:
    global WORKER_MODE
    WORKER_MODE = mode


def eval_busy() -> bool:
    return EVAL is not None and (EVAL.running or EVAL.building)


def public_scene_from_disk(slot: str) -> dict | None:
    try:
        path = slot_output_dir(slot) / "metadata.json"
    except RuntimeError:
        return None
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    return {
        "seed": meta.get("seed"),
        "instruction": meta.get("instruction"),
        "target": meta.get("target"),
        "distractors": meta.get("distractors", []),
        "room": meta.get("room"),
        "cameras": list(meta.get("cameras") or []),
    }


def slot_view(slot: str) -> dict:
    saved = public_scene_from_disk(slot) if slot != "grpo" else None
    scene = PREVIEW.scene(slot)
    if slot == "grpo":
        loaded = bool(GRPO is not None and GRPO.running and scene is not None)
    else:
        loaded = bool(worker_running() and scene is not None)
    return {
        "loaded": loaded,
        "scene": scene if loaded else None,
        "saved": saved,
        "gen": PREVIEW.scene_gen(slot),
        "frameGen": PREVIEW.frame_gen(slot),
    }


def worker_scene(slot: str) -> dict | None:
    return slot_view(slot)["scene"]


def generate_scene(
    seed: int | None = None,
    slot: str = "infer",
) -> dict:
    if GRPO is not None and GRPO.running:
        raise RuntimeError("GRPO is running")
    if eval_busy():
        raise RuntimeError("eval is running")
    config = load_config()
    if seed is not None:
        config["seed"] = int(seed)
        save_config(config)
    WORKER.ensure(config["compute"])
    reply = WORKER.request(
        {
            "op": "generate",
            "slot": slot,
            "seed": None if seed is None else int(seed),
            "output_dir": stored_path(slot_output_dir(slot)),
        },
        timeout=WORKER_GENERATE_S,
    )
    store_preview(slot, reply)
    VIEW_STATS.reset()
    set_worker_mode("shared")
    return reply["scene"]


def stop_infer_test() -> dict:
    if not worker_running():
        if TEST.snapshot().get("running"):
            snap = TEST.apply(
                {
                    "running": False,
                    "phase": "error",
                    "error": "render worker is not running",
                }
            )
            set_worker_mode("idle")
            return snap
        return TEST.reset_idle()
    try:
        reply = WORKER.request({"op": "test_stop"}, timeout=WORKER_STATUS_S)
        snap = TEST.apply(reply)
    except Exception as error:
        snap = TEST.apply(
            {
                "running": False,
                "phase": "error",
                "error": str(error),
            }
        )
    if not snap.get("running"):
        set_worker_mode("shared" if worker_running() else "idle")
    return snap


def reset_scene(slot: str = "infer") -> dict:
    if GRPO is not None and GRPO.running:
        raise RuntimeError("GRPO is running")
    if eval_busy():
        raise RuntimeError("eval is running")
    config = load_config()
    WORKER.ensure(config["compute"])
    reply = WORKER.request({"op": "reset", "slot": slot})
    store_preview(slot, reply)
    VIEW_STATS.reset()
    set_worker_mode("shared")
    return reply.get("scene") or worker_scene(slot) or {}


def build_snapshot(*, scan: bool = False, full: bool = False) -> dict:
    if not worker_running() and TEST.snapshot().get("running"):
        TEST.apply(
            {
                "running": False,
                "phase": "error",
                "error": "render worker exited",
            }
        )
        set_worker_mode("idle")
    if WORKER_MODE == "grpo" and (GRPO is None or not GRPO.running):
        set_worker_mode("idle")
        PREVIEW.clear_slot("grpo")
    if WORKER_MODE == "eval" and not eval_busy():
        set_worker_mode("idle")
    config = load_config()
    collect_view = slot_view("collect")
    infer_view = slot_view("infer")
    grpo_view = slot_view("grpo")
    hub = HUB_IMPORT.snapshot()
    payload = {
        "config": config,
        "devices": public_devices(probe=False),
        "modelDevices": public_model_devices(probe=False),
        "objects": object_catalog(config) if full else None,
        "worker": {
            "alive": worker_running(),
            "device": WORKER.device_id,
            "mode": WORKER_MODE,
        },
        "slots": {
            "collect": collect_view,
            "infer": infer_view,
            "grpo": grpo_view,
        },
        "collect": {
            **LEADER.status(scan=scan),
            "scene": collect_view["scene"],
            "record": COLLECT.snapshot(),
        },
        "infer": {
            "scene": infer_view["scene"],
            "checkpoints": list_checkpoints(),
        },
        "test": TEST.snapshot(),
        "train": TRAIN.snapshot(extras=False),
        "grpo": GRPO.snapshot(extras=False),
        "eval": EVAL.snapshot(),
        "hub": hub,
        "hubImport": hub,
        "stats": VIEW_STATS.snapshot(),
    }
    return payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        line = format % args
        if (
            "/api/stream" in line
            or "/api/stats" in line
            or "/api/view" in line
            or "/api/health" in line
            or "/api/runtime" in line
            or "/api/state" in line
        ):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), line))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = 200
        raw = self.headers.get("Range") or ""
        if raw.startswith("bytes=") and size:
            spec = raw.split("=", 1)[1].split(",")[0].strip()
            left, _, right = spec.partition("-")
            if left:
                start = max(0, min(size - 1, int(left)))
            if right:
                end = max(start, min(size - 1, int(right)))
            status = 206
        length = max(0, end - start + 1) if size else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            left = length
            while left:
                chunk = handle.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)
        self.wfile.flush()

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _repo_id(self, query=None, payload=None) -> str:
        if payload and payload.get("repo_id"):
            return str(payload["repo_id"])
        if query:
            value = str((query.get("repo_id") or [""])[0] or "")
            if value:
                return value
        return DATASET.repo_id

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/api/state":
            query = parse_qs(parsed.query)
            scan = str((query.get("scan") or [""])[0] or "") == "1"
            full = str((query.get("full") or [""])[0] or "") == "1"
            try:
                self._send_json(200, build_snapshot(scan=scan, full=full))
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/config":
            try:
                boot = load_config()
                boot["compute"]["model"] = public_model_device_id(
                    boot["compute"].get("model")
                )
                self._send_json(
                    200,
                    {
                        "config": boot,
                        "devices": public_devices(probe=False),
                        "modelDevices": public_model_devices(probe=False),
                    },
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/objects":
            self._send_json(200, {"objects": object_catalog(load_config())})
            return
        if path == "/api/objects/preview":
            query = parse_qs(parsed.query)
            name = str(query.get("id", [""])[0])
            try:
                self._send(200, object_preview(name, load_config()), "image/png")
            except Exception as error:
                self._send_json(404, {"error": str(error)})
            return
        if path == "/api/collect":
            query = parse_qs(parsed.query)
            scan = str((query.get("scan") or ["1"])[0] or "1") != "0"
            try:
                self._send_json(200, build_snapshot(scan=scan))
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/datasets":
            try:
                self._send_json(
                    200,
                    {
                        "datasets": DATASET.list_datasets(),
                        "repoId": DATASET.repo_id,
                        "hub": HUB_IMPORT.snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/episodes":
            try:
                repo_id = self._repo_id(parse_qs(parsed.query))
                self._send_json(
                    200,
                    {
                        "episodes": DATASET.list_episodes(repo_id),
                        "repoId": repo_id,
                    },
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path.startswith("/api/episodes/") and "/video/" in path:
            parts = path.strip("/").split("/")
            # api episodes {i_or_id} video {camera}
            if len(parts) != 5:
                self._send_json(404, {"error": "not found"})
                return
            try:
                repo_id = self._repo_id(parse_qs(parsed.query))
                video = DATASET.video_path(parts[2], parts[4], repo_id)
                if not video.is_file():
                    self._send_json(404, {"error": "not found"})
                    return
                self._send_file(video, "video/mp4")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception as error:
                self._send_json(404, {"error": str(error)})
            return
        if path == "/api/view":
            query = parse_qs(parsed.query)
            camera = str(query.get("camera", [""])[0])
            slot = str(query.get("slot", ["infer"])[0] or "infer")
            jpeg = PREVIEW.get(slot, camera)
            if not jpeg:
                self._send_json(404, {"error": "no frame"})
                return
            VIEW_STATS.add(camera)
            self._send(200, jpeg, "image/jpeg")
            return
        if path == "/api/stream":
            query = parse_qs(parsed.query)
            camera = str(query.get("camera", [""])[0])
            slot = str(query.get("slot", ["infer"])[0] or "infer")
            fps = int(query.get("fps", [15])[0] or 15)
            fps = max(1, min(STREAM_MAX_FPS, fps))
            interval = 1.0 / fps
            headers_sent = False
            try:
                self.connection.settimeout(None)
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
                )
                self.end_headers()
                headers_sent = True
                last = b""
                last_write = 0.0
                while True:
                    if slot in ("collect", "infer") and not worker_running():
                        time.sleep(interval)
                        continue
                    jpeg = PREVIEW.get(slot, camera)
                    now = time.monotonic()
                    if jpeg and (jpeg != last or now - last_write >= interval):
                        changed = jpeg != last
                        last = jpeg
                        last_write = now
                        if changed:
                            VIEW_STATS.add(camera)
                        self.wfile.write(
                            (
                                f"--{STREAM_BOUNDARY}\r\n"
                                "Content-Type: image/jpeg\r\n"
                                f"Content-Length: {len(jpeg)}\r\n\r\n"
                            ).encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(interval)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                return
            except Exception as error:
                if headers_sent:
                    return
                try:
                    self._send_json(409, {"error": str(error)})
                except Exception:
                    return
            return
        if path == "/api/stats":
            self._send_json(200, VIEW_STATS.snapshot())
            return
        if path == "/api/runtime":
            try:
                self._send_json(200, build_snapshot())
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/train":
            self._send_json(200, TRAIN.snapshot(extras=False))
            return
        if path == "/api/grpo":
            self._send_json(200, GRPO.snapshot(extras=False))
            return
        if path == "/api/eval":
            self._send_json(200, EVAL.snapshot())
            return
        if path.startswith("/api/eval/scene/") and path.endswith("/image"):
            try:
                parts = path.strip("/").split("/")
                index = int(parts[3])
                camera = str(parse_qs(urlparse(self.path).query).get("camera", ["cameras"])[0] or "cameras")
                if Path(camera).name != camera:
                    raise ValueError("invalid camera")
                dest = EVAL_VALSET_DIR / f"{index:04d}"
                if camera == "cameras":
                    file_path = dest / "cameras.png"
                else:
                    meta_path = dest / "metadata.json"
                    cameras = []
                    if meta_path.is_file():
                        meta = json.loads(meta_path.read_text())
                        cameras = list((meta or {}).get("cameras") or [])
                    if camera not in cameras:
                        raise FileNotFoundError(camera)
                    file_path = dest / f"{camera}.png"
                if not file_path.is_file():
                    raise FileNotFoundError(camera)
                self._send(200, file_path.read_bytes(), "image/png")
            except Exception as error:
                self._send_json(404, {"error": str(error)})
            return
        if path == "/api/checkpoints":
            self._send_json(200, {"checkpoints": list_checkpoints()})
            return
        if path == "/api/runs":
            self._send_json(200, {"runs": list_runs(_busy_runs())})
            return
        if path.startswith("/api/runs/"):
            name = unquote(path.rsplit("/", 1)[-1])
            try:
                self._send_json(200, {"run": load_run(name)})
            except Exception as error:
                self._send_json(404, {"error": str(error)})
            return
        if path == "/api/test":
            self._send_json(200, TEST.snapshot())
            return
        if path == "/":
            path = "/index.html"
        relative = path.lstrip("/")
        file_path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in file_path.parents and file_path != STATIC_DIR:
            self._send_json(403, {"error": "forbidden"})
            return
        if not file_path.is_file():
            self._send_json(404, {"error": "not found"})
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(file_path.suffix, "application/octet-stream")
        self._send(200, file_path.read_bytes(), content_type)

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/config":
            try:
                payload = self._read_json()
                path_keys = payload.get("path")
                if not path_keys:
                    raise ValueError("config patch requires path")
                if path_keys[0] == "compute":
                    probe = dict(load_config()["compute"])
                    if len(path_keys) > 1 and path_keys[1] == "render":
                        probe["render"] = payload.get("value")
                        resolve_render_device(probe)
                config = patch_config(path_keys, payload.get("value"))
                self._send_json(
                    200, {"ok": True, "config": config, "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path.startswith("/api/episodes/"):
            try:
                target = path.rsplit("/", 1)[-1]
                payload = self._read_json()
                repo_id = self._repo_id(payload=payload)
                DATASET.edit_task(target, str(payload.get("task") or ""), repo_id)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "episodes": DATASET.list_episodes(repo_id),
                        "repoId": repo_id,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path.startswith("/api/runs/"):
            name = unquote(path.rsplit("/", 1)[-1])
            try:
                delete_run(name, _busy_runs())
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "runs": list_runs(_busy_runs()),
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/datasets":
            try:
                repo_id = self._repo_id(query)
                snap = COLLECT.snapshot()
                if snap.get("recording") and snap.get("repoId") == repo_id:
                    raise RuntimeError("cannot delete the dataset while recording")
                if HUB_IMPORT.running and HUB_IMPORT.repo_id == repo_id:
                    raise RuntimeError("cannot delete the dataset while it is importing")
                DATASET.delete_dataset(repo_id)
                if COLLECT.repo_id == repo_id:
                    COLLECT.apply_settings({"repo_id": DATASET.repo_id})
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "datasets": DATASET.list_datasets(),
                        "repoId": DATASET.repo_id,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if not path.startswith("/api/episodes/"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            target = path.rsplit("/", 1)[-1]
            repo_id = self._repo_id(query)
            if DATASET.is_current(target, repo_id):
                raise RuntimeError("cannot delete the episode while it is recording")
            DATASET.delete_episode(target, repo_id)
            self._send_json(
                200,
                {
                    "ok": True,
                    "episodes": DATASET.list_episodes(repo_id),
                    "repoId": repo_id,
                    "state": build_snapshot(),
                },
            )
        except Exception as error:
            self._send_json(400, {"error": str(error)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/reset":
            try:
                payload = self._read_json()
                slot = str(payload.get("slot") or "infer")
                if GRPO is not None and GRPO.running:
                    raise RuntimeError("GRPO is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                if slot == "collect" and COLLECT.snapshot().get("recording"):
                    raise RuntimeError("cannot reset the scene while recording")
                if slot == "infer":
                    with pause_collect():
                        WORKER.ensure(load_config()["compute"])
                        stop_infer_test()
                        scene = reset_scene(slot)
                else:
                    scene = reset_scene(slot)
                if slot == "collect" and LEADER.status().get("step") == "ready":
                    COLLECT.start()
                self._send_json(
                    200, {"ok": True, "scene": scene, "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/generate":
            try:
                payload = self._read_json()
                seed = payload.get("seed")
                slot = str(payload.get("slot") or "infer")
                seed_n = None if seed is None else int(seed)
                if GRPO is not None and GRPO.running:
                    raise RuntimeError("GRPO is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                if slot == "collect" and COLLECT.snapshot().get("recording"):
                    raise RuntimeError("cannot generate the scene while recording")
                if slot == "infer":
                    with pause_collect():
                        WORKER.ensure(load_config()["compute"])
                        stop_infer_test()
                        scene = generate_scene(seed_n, slot=slot)
                else:
                    scene = generate_scene(seed_n, slot=slot)
                if slot == "collect" and LEADER.status().get("step") == "ready":
                    COLLECT.start()
                self._send_json(
                    200, {"ok": True, "scene": scene, "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/detect":
            sys.stderr.write("leader detect_start\n")
            sys.stderr.flush()
            try:
                self._send_json(
                    200, {"ok": True, **LEADER.detect_start(), "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/detect/unplug":
            try:
                self._send_json(
                    200, {"ok": True, **LEADER.detect_unplug(), "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/detect/replug":
            try:
                self._send_json(
                    200, {"ok": True, **LEADER.detect_replug(), "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/connect":
            try:
                self._send_json(
                    200, {"ok": True, **LEADER.connect(), "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/use_calibration":
            try:
                status = LEADER.use_saved()
                scene = generate_scene(slot="collect")
                COLLECT.start()
                self._send_json(
                    200,
                    {"ok": True, **status, "scene": scene, "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/calibrate/start":
            try:
                self._send_json(
                    200,
                    {"ok": True, **LEADER.start_calibrate(), "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/calibrate/home":
            try:
                status = LEADER.set_home()
                LEADER.start_range()
                self._send_json(
                    200, {"ok": True, **status, "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/leader/calibrate/finish":
            try:
                status = LEADER.finish_calibrate()
                scene = generate_scene(slot="collect")
                COLLECT.start()
                self._send_json(
                    200,
                    {"ok": True, **status, "scene": scene, "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/datasets":
            try:
                payload = self._read_json()
                repo_id = DATASET.create_dataset(str(payload.get("repo_id") or ""))
                COLLECT.apply_settings({"repo_id": repo_id})
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "datasets": DATASET.list_datasets(),
                        "repoId": repo_id,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/datasets/hub":
            try:
                payload = self._read_json()
                repo_id = str(payload.get("repo_id") or "").strip()
                snap = COLLECT.snapshot()
                if snap.get("recording") and snap.get("repoId") == repo_id:
                    raise RuntimeError("cannot import while recording this dataset")
                hub = HUB_IMPORT.start(repo_id)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "datasets": DATASET.list_datasets(),
                        "repoId": DATASET.repo_id,
                        "hub": hub,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/datasets/sync":
            try:
                payload = self._read_json()
                repo_id = str(payload.get("repo_id") or DATASET.repo_id).strip()
                snap = COLLECT.snapshot()
                if snap.get("recording") and snap.get("repoId") == repo_id:
                    raise RuntimeError("cannot sync while recording this dataset")
                hub = HUB_IMPORT.start_sync(repo_id)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "datasets": DATASET.list_datasets(),
                        "repoId": DATASET.repo_id,
                        "hub": hub,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/datasets/select":
            try:
                payload = self._read_json()
                repo_id = str(payload.get("repo_id") or "").strip() or DEFAULT_REPO_ID
                DATASET.repo_id = repo_id
                COLLECT.apply_settings({"repo_id": repo_id})
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "datasets": DATASET.list_datasets(),
                        "repoId": repo_id,
                        "episodes": DATASET.list_episodes(repo_id),
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/collect/settings":
            try:
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "record": COLLECT.apply_settings(self._read_json()),
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/collect/record":
            try:
                if GRPO is not None and GRPO.running:
                    raise RuntimeError("GRPO is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                payload = self._read_json()
                scene = worker_scene("collect")
                if scene is None:
                    raise RuntimeError("create a collect scene first")
                reply = COLLECT.begin_record(
                    payload,
                    list(scene.get("cameras") or []),
                    str(payload.get("task") or scene.get("instruction") or ""),
                )
                self._send_json(
                    200, {"ok": True, "record": reply, "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/collect/record/stop":
            try:
                payload = self._read_json()
                save = bool(payload.get("save", True))
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "record": COLLECT.end_record(save),
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/test":
            try:
                if GRPO is not None and GRPO.running:
                    raise RuntimeError("GRPO is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                payload = self._read_json()
                config = load_config()
                WORKER.ensure(config["compute"])
                reply = WORKER.request(
                    {
                        "op": "test_start",
                        "duration_seconds": float(
                            payload.get("duration_seconds") or 20
                        ),
                        "n_action_steps": int(
                            payload.get("n_action_steps") or 50
                        ),
                        "num_steps": int(payload.get("num_steps") or 10),
                        "control_hz": float(payload.get("control_hz") or 15),
                        "fps": int(payload.get("fps") or 15),
                        "checkpoint": str(
                            payload.get("checkpoint") or "lerobot/smolvla_base"
                        ),
                        "cameras": payload_cameras(payload)
                        or [
                            str(name)
                            for name in (
                                (config.get("compute") or {}).get("policy_cameras")
                                or []
                            )
                            if str(name).strip()
                        ],
                        "policy_mode": str(
                            payload.get("policy_mode")
                            or config.get("policy_mode")
                            or "smolvla"
                        ),
                    }
                )
                snap = TEST.apply(reply)
                set_worker_mode("infer_test")
                self._send_json(200, {**snap, "state": build_snapshot()})
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/test/stop":
            self._send_json(200, {**stop_infer_test(), "state": build_snapshot()})
            return
        if path == "/api/train":
            try:
                if GRPO.running:
                    raise RuntimeError("GRPO is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                payload = self._read_json()
                config = load_config()
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "train": TRAIN.start(
                            payload,
                            DATASET,
                            config["compute"],
                            config["training"],
                        ),
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/train/stop":
            try:
                self._send_json(
                    200, {"ok": True, "train": TRAIN.stop(), "state": build_snapshot()}
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/grpo":
            try:
                if TRAIN.running:
                    raise RuntimeError("SFT training is running")
                if eval_busy():
                    raise RuntimeError("eval is running")
                payload = self._read_json()
                config = load_config()
                COLLECT.stop_join()
                TEST.reset_idle()
                grpo = GRPO.start(
                    payload,
                    config["compute"],
                    config,
                    PREVIEW,
                    WORKER,
                )
                set_worker_mode("grpo")
                PREVIEW.clear_sim_slots()
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "grpo": grpo,
                        "state": build_snapshot(),
                    },
                )
            except Exception as error:
                if GRPO is None or not GRPO.running:
                    if WORKER_MODE == "grpo":
                        set_worker_mode("idle" if not worker_running() else "shared")
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/grpo/stop":
            try:
                self._send_json(
                    200,
                    {"ok": True, "grpo": GRPO.stop_run(), "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/eval":
            try:
                if TRAIN.running:
                    raise RuntimeError("SFT training is running")
                if GRPO.running:
                    raise RuntimeError("GRPO is running")
                if TEST.snapshot().get("running"):
                    raise RuntimeError("inference test is running")
                payload = self._read_json()
                config = load_config()
                COLLECT.stop_join()
                TEST.reset_idle()
                ev = EVAL.start(
                    payload,
                    config["compute"],
                    config,
                    WORKER,
                )
                set_worker_mode("eval")
                PREVIEW.clear_sim_slots()
                self._send_json(
                    200,
                    {"ok": True, "eval": ev, "state": build_snapshot()},
                )
            except Exception as error:
                if not eval_busy():
                    if WORKER_MODE == "eval":
                        set_worker_mode("idle" if not worker_running() else "shared")
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/eval/stop":
            try:
                self._send_json(
                    200,
                    {"ok": True, "eval": EVAL.stop_run(), "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)})
            return
        if path == "/api/eval/dataset":
            try:
                if TRAIN.running:
                    raise RuntimeError("SFT training is running")
                if GRPO.running:
                    raise RuntimeError("GRPO is running")
                if TEST.snapshot().get("running"):
                    raise RuntimeError("inference test is running")
                payload = self._read_json()
                config = load_config()
                COLLECT.stop_join()
                ev = EVAL.start_build(payload, config["compute"], WORKER)
                set_worker_mode("eval")
                PREVIEW.clear_sim_slots()
                self._send_json(
                    200,
                    {"ok": True, "eval": ev, "state": build_snapshot()},
                )
            except Exception as error:
                if not eval_busy():
                    if WORKER_MODE == "eval":
                        set_worker_mode("idle" if not worker_running() else "shared")
                self._send_json(400, {"error": str(error)})
            return
        if path == "/api/eval/dataset/reroll":
            try:
                if TRAIN.running:
                    raise RuntimeError("SFT training is running")
                if GRPO.running:
                    raise RuntimeError("GRPO is running")
                if TEST.snapshot().get("running"):
                    raise RuntimeError("inference test is running")
                payload = self._read_json()
                config = load_config()
                COLLECT.stop_join()
                ev = EVAL.reroll(payload, config["compute"], WORKER)
                set_worker_mode("idle")
                self._send_json(
                    200,
                    {"ok": True, "eval": ev, "state": build_snapshot()},
                )
            except Exception as error:
                self._send_json(400, {"error": str(error)})
            return
        self._send_json(404, {"error": "not found"})


def run_worker() -> None:
    import io
    import shutil
    import time

    import mujoco
    import numpy as np
    from PIL import Image

    from cameras.cameras import SharedRenderer, apply_camera_pipeline
    from scene.generate import generate

    sys.stdout = sys.stderr
    out = open(1, "wb", closefd=False)

    def encode_jpeg(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=VIEW_JPEG_QUALITY)
        return buffer.getvalue()

    def open_ffmpeg(path: Path, fps: int, width: int, height: int) -> subprocess.Popen:
        return subprocess.Popen(
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
                f"{int(width)}x{int(height)}",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                FFMPEG_PRESET,
                "-crf",
                str(FFMPEG_CRF),
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    class EpisodeRecorder:
        def __init__(self) -> None:
            self.queue: queue.Queue = queue.Queue(maxsize=RECORD_FRAME_QUEUE)
            self.thread: threading.Thread | None = None
            self.writers: dict[str, subprocess.Popen] = {}
            self.viewer = None
            self.cameras: list[str] = []
            self.frames = 0
            self.dropped = 0
            self.error: str | None = None
            self.stopping = threading.Event()

        def start(
            self,
            folder: Path,
            cameras: list[str],
            fps: int,
            width: int,
            height: int,
            viewer,
        ) -> None:
            self.stop(save=False)
            self.queue = queue.Queue(maxsize=RECORD_FRAME_QUEUE)
            self.cameras = list(cameras)
            self.viewer = viewer
            self.frames = 0
            self.dropped = 0
            self.error = None
            self.stopping.clear()
            self.writers = {
                name: open_ffmpeg(folder / f"{name}.mp4", fps, width, height)
                for name in self.cameras
            }
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        def capture(self, viewer) -> dict:
            if not self.cameras or self.stopping.is_set():
                return {}
            if self.error:
                raise RuntimeError(self.error)
            frames = {name: viewer.record_rgb(name) for name in self.cameras}
            if self.writers:
                self.queue.put(frames, timeout=FFMPEG_WAIT_S)
            return frames

        def _loop(self) -> None:
            try:
                while True:
                    try:
                        item = self.queue.get(timeout=0.05)
                    except queue.Empty:
                        if self.stopping.is_set():
                            break
                        continue
                    if item is None:
                        break
                    for name, pixels in item.items():
                        proc = self.writers.get(name)
                        if proc is None or proc.stdin is None or proc.poll() is not None:
                            raise RuntimeError(f"ffmpeg {name} exited")
                        image = (
                            self.viewer.apply_defects(
                                pixels, name, frame_id=self.frames
                            )
                            if self.viewer is not None
                            else Image.fromarray(pixels)
                        )
                        rgb = np.ascontiguousarray(
                            np.asarray(image.convert("RGB"), dtype=np.uint8)
                        )
                        proc.stdin.write(rgb.tobytes())
                    self.frames += 1
            except Exception as error:
                if not self.stopping.is_set():
                    self.error = str(error)

        def _drain_queue(self) -> None:
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        def _close_stdins(self) -> None:
            for proc in self.writers.values():
                if proc.stdin is None:
                    continue
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        def _wait_writers(self, timeout: float = FFMPEG_WAIT_S) -> None:
            deadline = time.monotonic() + timeout
            for proc in self.writers.values():
                try:
                    proc.wait(timeout=max(0.05, deadline - time.monotonic()))
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            for proc in self.writers.values():
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass

        def _kill_writers(self) -> None:
            for proc in self.writers.values():
                try:
                    proc.kill()
                except Exception:
                    pass

        def stop(self, save: bool) -> tuple[int, int, bool]:
            thread = self.thread
            forced = False
            self.stopping.set()
            if not save:
                self._drain_queue()
                self._kill_writers()
            if thread is not None and thread.is_alive():
                thread.join(timeout=FFMPEG_JOIN_S if save else 2.0)
                if thread.is_alive():
                    self._kill_writers()
                    thread.join(timeout=3.0)
                    forced = True
            self._close_stdins()
            if save and not forced:
                self._wait_writers()
            else:
                self._kill_writers()
                self._wait_writers(timeout=2.0)
            frames = self.frames
            dropped = self.dropped
            error = self.error
            alive = thread is not None and thread.is_alive()
            if not alive:
                self.writers = {}
                self.thread = None
                self.viewer = None
                self.cameras = []
            if alive:
                raise RuntimeError("video encoder did not stop")
            if error and save and not forced:
                raise RuntimeError(error)
            return frames, dropped, forced

    class SceneViewer:
        def __init__(self) -> None:
            self.model = None
            self.data = None
            self.view_data = None
            self.renderer = None
            self.preview_renderer = None
            self.metadata: dict | None = None
            self.config: dict | None = None
            self.camera_names: list[str] = []
            self.camera_profiles: dict[str, dict] = {}
            self._jpeg_cache: dict[str, bytes] = {}
            self._applied_gen = -1
            self.ctrl_gen = 0
            self.control_hz = 15.0
            self.physics_steps = 1

        def close(self) -> None:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            if self.preview_renderer is not None:
                self.preview_renderer.close()
                self.preview_renderer = None
            self.model = None
            self.data = None
            self.view_data = None
            self.camera_names = []
            self.camera_profiles = {}
            self._jpeg_cache = {}
            self._applied_gen = -1
            self.ctrl_gen = 0

        def load(self, scene_dir: Path) -> dict:
            metadata_path = scene_dir / "metadata.json"
            xml_path = scene_dir / "scene.xml"
            if not xml_path.is_file():
                raise FileNotFoundError(f"no scene at {scene_dir}")
            self.close()
            self.config = load_config()
            self.metadata = json.loads(metadata_path.read_text())
            self.camera_names = [
                str(name) for name in self.metadata.get("cameras") or []
            ]
            self.camera_profiles = dict(
                self.metadata.get("camera_profiles") or {}
            )
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
            self.data = mujoco.MjData(self.model)
            self.view_data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_forward(self.model, self.view_data)
            self.renderer = SharedRenderer(
                self.model,
                height=VIEW_SIZE,
                width=VIEW_SIZE,
            )
            self.preview_renderer = SharedRenderer(
                self.model,
                height=VIEW_SIZE,
                width=VIEW_SIZE,
            )
            self.control_hz = float(
                (self.config.get("environment") or {})
                .get("rollout", {})
                .get("control_hz")
                or 15
            )
            self.physics_steps = max(
                1,
                round((1.0 / self.control_hz) / self.model.opt.timestep),
            )
            self._jpeg_cache = {}
            self.reset_home()
            return self.public_metadata()

        def public_metadata(self) -> dict | None:
            if self.metadata is None:
                return None
            return {
                "seed": self.metadata.get("seed"),
                "instruction": self.metadata.get("instruction"),
                "target": self.metadata.get("target"),
                "distractors": self.metadata.get("distractors", []),
                "room": self.metadata.get("room"),
                "cameras": list(self.camera_names),
            }

        def apply_defects(
            self, pixels, camera_name: str, frame_id: int | None = None
        ) -> Image.Image:
            profile = self.camera_profiles.get(camera_name)
            if profile is None or self.config is None or self.metadata is None:
                return Image.fromarray(pixels)
            seed = (
                int(self.metadata.get("seed") or 0) * 10007
                + sum(ord(char) for char in camera_name)
                + int(frame_id or self.ctrl_gen) * 65537
            ) & 0xFFFFFFFF
            processed, _ = apply_camera_pipeline(
                pixels,
                self.config,
                np.random.default_rng(seed),
                profile,
            )
            return processed

        def render(self, camera_name: str, data=None) -> bytes:
            if self.model is None or self.preview_renderer is None:
                raise RuntimeError("no scene loaded")
            if camera_name not in self.camera_names:
                raise RuntimeError(f"unknown camera {camera_name}")
            source = self.view_data if data is None and self.view_data is not None else (
                data if data is not None else self.data
            )
            with self.preview_renderer.lock:
                pixels = self.preview_renderer.render_rgb(
                    source, camera_name, copy=True
                )
            jpeg = encode_jpeg(self.apply_defects(pixels, camera_name))
            self._jpeg_cache[camera_name] = jpeg
            return jpeg

        def apply_view(self, qpos, qvel) -> None:
            if self.model is None or self.view_data is None:
                return
            self.view_data.qpos[: len(qpos)] = qpos
            self.view_data.qvel[: len(qvel)] = qvel
            mujoco.mj_forward(self.model, self.view_data)
            self._jpeg_cache = {}

        def apply_state(self, qpos, qvel) -> None:
            if self.model is None or self.data is None:
                return
            self.data.qpos[: len(qpos)] = qpos
            self.data.qvel[: len(qvel)] = qvel
            mujoco.mj_forward(self.model, self.data)
            self.apply_view(qpos, qvel)

        def reset_home(self) -> None:
            if self.model is None or self.data is None or self.config is None:
                raise RuntimeError("no scene loaded")
            mujoco.mj_resetData(self.model, self.data)
            home_key = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            if home_key >= 0:
                self.data.qpos[:ROBOT_ACTION_DIM] = self.model.key_qpos[
                    home_key, :ROBOT_ACTION_DIM
                ]
                self.data.ctrl[:ROBOT_ACTION_DIM] = self.model.key_ctrl[
                    home_key, :ROBOT_ACTION_DIM
                ]
            mujoco.mj_forward(self.model, self.data)
            settle = round(
                float(self.config["scene"]["settle_seconds"])
                / self.model.opt.timestep
            )
            if settle:
                mujoco.mj_step(self.model, self.data, nstep=settle)
            self.ctrl_gen += 1
            self.apply_view(self.data.qpos, self.data.qvel)

        def _home_ctrl(self) -> np.ndarray:
            home_key = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            if home_key < 0:
                return np.zeros(ROBOT_ACTION_DIM, dtype=np.float64)
            return np.asarray(
                self.model.key_ctrl[home_key, :ROBOT_ACTION_DIM],
                dtype=np.float64,
            )

        def apply_ctrl(self, action_deg, hz=None) -> list[float]:
            if self.model is None or self.data is None:
                raise RuntimeError("no scene loaded")
            action = np.deg2rad(np.asarray(action_deg, dtype=np.float64))
            if action.shape != (ROBOT_ACTION_DIM,):
                raise RuntimeError("bad action shape")
            if hz:
                self.physics_steps = max(
                    1,
                    round((1.0 / float(hz)) / self.model.opt.timestep),
                )
            limits = self.model.actuator_ctrlrange[:ROBOT_ACTION_DIM]
            home = self._home_ctrl().copy()
            home[WRIST_FLEX_INDEX] = 0.0
            action = action + home
            gripper_fraction = np.clip(
                (float(action_deg[GRIPPER_INDEX]) - LEADER_GRIPPER_MIN_DEG)
                / (LEADER_GRIPPER_MAX_DEG - LEADER_GRIPPER_MIN_DEG),
                0.0,
                1.0,
            )
            action[GRIPPER_INDEX] = (
                limits[GRIPPER_INDEX, 0]
                + gripper_fraction
                * (limits[GRIPPER_INDEX, 1] - limits[GRIPPER_INDEX, 0])
            )
            action = np.clip(action, limits[:, 0], limits[:, 1])
            self.data.ctrl[:ROBOT_ACTION_DIM] = action
            mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
            self.ctrl_gen += 1
            self.apply_view(self.data.qpos, self.data.qvel)
            return np.rad2deg(
                self.data.qpos[:ROBOT_ACTION_DIM]
            ).astype(np.float32).tolist()

        def record_rgb(self, camera_name: str, data=None):
            if self.model is None or self.renderer is None:
                raise RuntimeError("no scene loaded")
            source = self.view_data if data is None and self.view_data is not None else (
                data if data is not None else self.data
            )
            with self.renderer.lock:
                return self.renderer.render_rgb(source, camera_name, copy=True)

        def encode_cameras(self, names: list[str], data=None) -> list[bytes]:
            return [self._encode_raw(name, data) for name in names]

        def preview_jpegs(self, data=None) -> tuple[list[str], list[bytes]]:
            names = list(self.camera_names)
            return names, [self.render(name, data) for name in names]

        def _encode_raw(self, camera_name: str, data=None) -> bytes:
            if self.model is None or self.renderer is None:
                raise RuntimeError("no scene loaded")
            source = self.view_data if data is None and self.view_data is not None else (
                data if data is not None else self.data
            )
            with self.renderer.lock:
                pixels = self.renderer.render_rgb(source, camera_name, copy=True)
            return encode_jpeg(self.apply_defects(pixels, camera_name))

        def sync_live(self, session: "TestSession") -> None:
            with session.lock:
                packed = session.latest_state
            if packed is None:
                return
            qpos, qvel, gen = packed
            if gen == self._applied_gen:
                return
            self.apply_state(qpos, qvel)
            self._applied_gen = gen
            for name in self.camera_names:
                self._jpeg_cache[name] = self._encode_raw(name)

    class TestSession:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.stop = threading.Event()
            self.thread: threading.Thread | None = None
            self.frames: dict[str, bytes] = {}
            self.logs: list[dict] = []
            self.llm_log_offset = 0
            self.running = False
            self.latest_state = None
            self.state_gen = 0
            self.status = {
                "running": False,
                "phase": "idle",
                "seconds": 0.0,
                "success": None,
                "error": None,
                "logs": [],
            }

        def snapshot(self) -> dict:
            with self.lock:
                payload = dict(self.status)
                payload["logs"] = list(self.logs)
                return payload

        def log(self, text: str, kind: str = "info") -> None:
            with self.lock:
                self.logs.append({"kind": kind, "text": text})
                self.logs = self.logs[-200:]
                self.status["logs"] = list(self.logs)
                payload = dict(self.status)
                payload["logs"] = list(self.logs)
            emit = getattr(self, "emit", None)
            if emit is not None:
                emit(payload)

        def pull_llm_logs(self) -> None:
            path = LLM_LOG_PATH
            if not path.is_file():
                return
            data = path.read_bytes()
            if len(data) <= self.llm_log_offset:
                return
            chunk = data[self.llm_log_offset :]
            self.llm_log_offset = len(data)
            for raw in chunk.decode("utf-8", "replace").splitlines():
                if not raw.strip():
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    self.log(raw)
                    continue
                latency = item.get("latency_ms")
                cameras = ",".join(item.get("cameras") or [])
                self.log(
                    f"Infer {item.get('num_steps')} steps"
                    + (f", {latency} ms" if latency is not None else "")
                    + (f", {cameras}" if cameras else "")
                )

        def frame(self, camera_name: str) -> bytes | None:
            with self.lock:
                return self.frames.get(camera_name)

        def set_status(self, **values) -> None:
            with self.lock:
                self.status.update(values)
                if "running" in values:
                    self.running = bool(values["running"])
                payload = dict(self.status)
                payload["logs"] = list(self.logs)
            emit = getattr(self, "emit", None)
            if emit is not None:
                emit(payload)

        def set_frames(self, frames: dict) -> None:
            encoded = {
                name: encode_jpeg(image) for name, image in frames.items()
            }
            with self.lock:
                self.frames = encoded
                self.status["seconds"] = float(
                    self.status.get("seconds") or 0.0
                )

        def set_state(self, qpos, qvel) -> None:
            with self.lock:
                self.state_gen += 1
                self.latest_state = (qpos, qvel, self.state_gen)

        def stop_join(self, pump: bool = False) -> None:
            self.stop.set()
            self.set_status(running=False, phase="idle")
            thread = self.thread
            deadline = time.monotonic() + 8
            while thread is not None and thread.is_alive() and time.monotonic() < deadline:
                if pump:
                    drain_obs_jobs()
                thread.join(timeout=0.05)
            fail_obs_jobs()
            if self.thread is thread:
                self.thread = None

    viewers = {"collect": SceneViewer(), "infer": SceneViewer()}
    obs_jobs: queue.Queue = queue.Queue()

    class ProxyRenderer:
        def __init__(self, height: int, width: int) -> None:
            self.lock = threading.Lock()
            self.height = int(height)
            self.width = int(width)

        def render_rgb(self, data, camera_name: str, copy: bool = False):
            waiter: queue.Queue = queue.Queue(maxsize=1)
            obs_jobs.put(
                (
                    np.array(data.qpos, copy=True),
                    np.array(data.qvel, copy=True),
                    str(camera_name),
                    self.height,
                    self.width,
                    waiter,
                )
            )
            try:
                pixels = waiter.get(timeout=OBS_RENDER_S)
            except queue.Empty:
                raise RuntimeError("infer render timed out")
            if pixels is None:
                raise RuntimeError("infer render failed")
            return pixels

        def close(self) -> None:
            return

    def drain_obs_jobs() -> None:
        viewer = viewers.get("infer")
        while True:
            try:
                qpos, qvel, camera_name, height, width, waiter = obs_jobs.get_nowait()
            except queue.Empty:
                return
            try:
                if (
                    viewer is None
                    or viewer.model is None
                    or viewer.view_data is None
                    or viewer.renderer is None
                ):
                    raise RuntimeError("no scene loaded")
                viewer.apply_view(qpos, qvel)
                with viewer.renderer.lock:
                    pixels = viewer.renderer.render_rgb(
                        viewer.view_data,
                        camera_name,
                        copy=True,
                    )
                if pixels.shape[0] != height or pixels.shape[1] != width:
                    pixels = np.asarray(
                        Image.fromarray(pixels).resize(
                            (width, height),
                            Image.BILINEAR,
                        )
                    )
                waiter.put(pixels)
            except Exception:
                waiter.put(None)

    def fail_obs_jobs() -> None:
        while True:
            try:
                *_, waiter = obs_jobs.get_nowait()
            except queue.Empty:
                return
            try:
                waiter.put(None)
            except Exception:
                pass

    session = TestSession()
    req_id_box = {"id": None}
    reply_lock = threading.Lock()

    def get_viewer(slot: str) -> SceneViewer:
        if slot not in viewers:
            raise RuntimeError(f"unknown slot {slot}")
        return viewers[slot]

    def reply(
        payload: dict,
        blobs: list[bytes] | None = None,
        req_id=None,
        event: bool = False,
    ) -> None:
        blobs = blobs or []
        payload = dict(payload)
        if req_id is None and not event:
            req_id = req_id_box["id"]
        if req_id is not None:
            payload["id"] = req_id
        if blobs:
            payload["blobs"] = [len(item) for item in blobs]
        with reply_lock:
            out.write((json.dumps(payload) + "\n").encode("utf-8"))
            for item in blobs:
                out.write(item)
            out.flush()

    session.emit = lambda payload: reply(
        {"event": "test_status", **payload},
        event=True,
    )

    class RenderHub:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.jobs: deque = deque()
            self.preview = {"collect": ([], []), "infer": ([], [])}
            self.record = {"collect": {}, "infer": {}}
            self.encode_waiters: dict[tuple[str, int], int] = {}

        def publish(self, slot: str, viewer: "SceneViewer", record: bool = False) -> int:
            if viewer.data is None:
                return int(viewer.ctrl_gen)
            qpos = np.array(viewer.data.qpos, copy=True)
            qvel = np.array(viewer.data.qvel, copy=True)
            return self.publish_arrays(slot, qpos, qvel, int(viewer.ctrl_gen), record)

        def publish_arrays(
            self,
            slot: str,
            qpos,
            qvel,
            gen: int,
            record: bool = False,
        ) -> int:
            job = (slot, qpos, qvel, int(gen), bool(record))
            fails = []
            with self.lock:
                if not record:
                    self.jobs = deque(
                        item
                        for item in self.jobs
                        if not (item[0] == slot and not item[4])
                    )
                self.jobs.append(job)
                while len(self.jobs) > RECORD_QUEUE_MAX:
                    fails.append(self.jobs.popleft())
            for dropped in fails:
                self._fail_job(dropped, f"dropped record gen {dropped[3]}")
            return int(gen)

        def clear_slot(self, slot: str) -> None:
            fails = []
            waiters = []
            with self.lock:
                kept = deque()
                for item in self.jobs:
                    if item[0] == slot:
                        fails.append(item)
                    else:
                        kept.append(item)
                self.jobs = kept
                self.record[slot].clear()
                self.preview[slot] = ([], [])
                for key in list(self.encode_waiters):
                    if key[0] == slot:
                        waiters.append(self.encode_waiters.pop(key))
            for item in fails:
                self._fail_job(item, "scene reloaded")
            for req_id in waiters:
                reply({"ok": False, "error": "scene reloaded"}, req_id=req_id)

        def set_preview(self, slot: str, names: list[str], blobs: list[bytes]) -> None:
            with self.lock:
                self.preview[slot] = (list(names), list(blobs))

        def preview_copy(self, slot: str) -> tuple[list[str], list[bytes]]:
            with self.lock:
                names, blobs = self.preview.get(slot, ([], []))
                return list(names), list(blobs)

        def take_or_wait(self, slot: str, gen: int, req_id) -> tuple | None:
            with self.lock:
                packed = self.record[slot].pop(int(gen), None)
                if packed is not None:
                    return packed
                if req_id is not None:
                    self.encode_waiters[(slot, int(gen))] = int(req_id)
                return None

        def _fail_job(self, job, error: str) -> None:
            slot, _qpos, _qvel, gen, record = job
            if not record:
                return
            with self.lock:
                req_id = self.encode_waiters.pop((slot, int(gen)), None)
            if req_id is not None:
                reply({"ok": False, "error": error}, req_id=req_id)

        def step(self) -> tuple[bool, bool]:
            with self.lock:
                if not self.jobs:
                    return False, False
                slot, qpos, qvel, gen, record = self.jobs.popleft()
            viewer = viewers.get(slot)
            if viewer is None or viewer.model is None or viewer.view_data is None:
                return True, False
            recorded = False
            try:
                viewer.apply_view(qpos, qvel)
                if record:
                    frames = recorder.capture(viewer)
                    recorded = bool(frames)
                    if frames:
                        names = list(frames)
                        blobs = [
                            encode_jpeg(Image.fromarray(frames[name]))
                            for name in names
                        ]
                    else:
                        names, blobs = viewer.preview_jpegs()
                else:
                    names, blobs = viewer.preview_jpegs()
                with self.lock:
                    self.preview[slot] = (names, blobs)
                    if record:
                        packed = (list(names), list(blobs))
                        self.record[slot][int(gen)] = packed
                        waiter_id = self.encode_waiters.pop((slot, int(gen)), None)
                    else:
                        waiter_id = None
                if slot == "infer" and not record:
                    reply(
                        {"event": "preview", "slot": slot, "cameras": names},
                        blobs=blobs,
                        event=True,
                    )
                if waiter_id is not None:
                    reply(
                        {"ok": True, "cameras": packed[0]},
                        blobs=packed[1],
                        req_id=waiter_id,
                    )
            except Exception as error:
                with self.lock:
                    waiter_id = self.encode_waiters.pop((slot, int(gen)), None)
                if waiter_id is not None:
                    reply({"ok": False, "error": str(error)}, req_id=waiter_id)
            return True, recorded

    hub = RenderHub()
    recorder = EpisodeRecorder()
    incoming: queue.Queue = queue.Queue()

    def stdin_reader() -> None:
        for raw in sys.stdin:
            incoming.put(json.loads(raw))
        incoming.put(None)

    threading.Thread(target=stdin_reader, daemon=True).start()

    def push_infer_preview(names: list[str], blobs: list[bytes]) -> None:
        if not names or len(names) != len(blobs):
            return
        reply(
            {"event": "preview", "slot": "infer", "cameras": names},
            blobs=blobs,
            event=True,
        )

    def run_test_thread(message: dict) -> None:
        try:
            with session.lock:
                session.logs = []
                session.status["logs"] = []
            session.llm_log_offset = 0
            session.latest_state = None
            session.state_gen = 0
            infer_viewer = viewers["infer"]
            infer_viewer._applied_gen = -1
            with hub.lock:
                hub.jobs = deque(
                    item for item in hub.jobs if item[0] != "infer"
                )
            session.set_status(
                running=True,
                phase="loading",
                seconds=0.0,
                success=None,
                error=None,
            )
            config = load_config()
            import torch
            import model.inference as inference

            mode = str(
                message.get("policy_mode")
                or config.get("policy_mode")
                or "smolvla"
            )
            requested = str(config.get("compute", {}).get("model") or "auto")
            wanted = resolve_model_device(requested)
            if requested not in ("auto", "cpu") and wanted == "cpu":
                session.log("CUDA is not available, using CPU.")
            if mode != "act":
                engine = inference.get_engine()
                current = str(engine.device)
                if current == "cuda":
                    current = f"cuda:{torch.cuda.current_device()}"
                if current != wanted:
                    session.log(f"Reload model {current} → {wanted}")
                    inference._ENGINE = inference.SmolVLAEngine(device=wanted)
            control_hz = float(message.get("control_hz") or 15)
            config["environment"]["rollout"]["control_hz"] = control_hz
            session.log(
                f"Device {wanted}, control {control_hz:g} Hz, "
                f"view {int(message.get('fps') or 15)} FPS, "
                f"checkpoint {message.get('checkpoint') or ('' if mode == 'act' else inference.MODEL_ID)}."
            )
            if infer_viewer.metadata and infer_viewer.metadata.get("instruction"):
                session.log(str(infer_viewer.metadata["instruction"]))
            if infer_viewer.model is None or infer_viewer.renderer is None:
                raise RuntimeError("no scene loaded")

            def on_frames(frames: dict) -> None:
                session.set_frames(frames)
                with session.lock:
                    names = list(session.frames)
                    blobs = [session.frames[name] for name in names]
                push_infer_preview(names, blobs)
                session.set_status(phase="running")

            def on_state(qpos, qvel) -> None:
                session.set_state(qpos, qvel)
                with session.lock:
                    gen = session.state_gen
                hub.publish_arrays("infer", qpos, qvel, gen, record=False)

            from core.environment import RandomSceneEnv

            orig_env_init = RandomSceneEnv.__init__
            rollout = config.get("environment", {}).get("rollout") or {}
            proxy_renderer = ProxyRenderer(
                int(rollout.get("camera_height") or VIEW_SIZE),
                int(rollout.get("camera_width") or VIEW_SIZE),
            )

            def reuse_viewer_env(self, scene_dir, config, sensor_seed, model=None, renderer=None):
                orig_env_init(
                    self,
                    scene_dir,
                    config,
                    sensor_seed,
                    model=None,
                    renderer=proxy_renderer,
                )

            RandomSceneEnv.__init__ = reuse_viewer_env
            try:
                if mode == "act":
                    from model.act import run_act_test

                    result = run_act_test(
                        UI_SCENE_DIR,
                        config,
                        float(message["duration_seconds"]),
                        int(message["n_action_steps"]),
                        on_frames,
                        session.stop.is_set,
                        session.log,
                        view_fps=float(message.get("fps") or 15),
                        on_state=on_state,
                        checkpoint=str(message.get("checkpoint") or ""),
                        cameras=list(message.get("cameras") or []),
                    )
                else:
                    result = inference.run_smolvla_test(
                        UI_SCENE_DIR,
                        config,
                        float(message["duration_seconds"]),
                        int(message["n_action_steps"]),
                        int(message["num_steps"]),
                        on_frames,
                        session.stop.is_set,
                        session.log,
                        view_fps=float(message.get("fps") or 15),
                        on_state=on_state,
                        checkpoint=str(message.get("checkpoint") or inference.MODEL_ID),
                        cameras=list(message.get("cameras") or []),
                    )
            finally:
                RandomSceneEnv.__init__ = orig_env_init
            session.set_state(result["qpos"], result["qvel"])
            with session.lock:
                gen = session.state_gen
            hub.publish_arrays(
                "infer",
                result["qpos"],
                result["qvel"],
                gen,
                record=False,
            )
            kind = "ok" if result["success"] else "info"
            session.log(
                (
                    "Success"
                    if result["success"]
                    else "Finished without success"
                )
                + f" in {float(result['seconds']):.1f}s, {result['steps']} steps.",
                kind,
            )
            session.set_status(
                running=False,
                phase="done",
                seconds=float(result["seconds"]),
                success=bool(result["success"]),
                error=None,
            )
        except Exception as error:
            session.log(str(error), "error")
            session.set_status(
                running=False,
                phase="error",
                error=str(error),
            )

    def handle(message: dict) -> None:
        req_id_box["id"] = message.get("id")
        op = message.get("op")
        if op == "scene":
            slot = str(message.get("slot") or "infer")
            reply({"ok": True, "scene": get_viewer(slot).public_metadata()})
        elif op == "generate":
            slot = str(message.get("slot") or "infer")
            output_dir = resolve_stored(
                message.get("output_dir") or stored_path(slot_output_dir(slot))
            )
            viewer = get_viewer(slot)
            if slot == "infer":
                session.stop_join(pump=True)
            if slot == "collect":
                recorder.stop(save=False)
            hub.clear_slot(slot)
            viewer.close()
            generate(
                seed_override=message.get("seed"),
                output_dir_override=output_dir,
            )
            scene = viewer.load(output_dir)
            names, blobs = viewer.preview_jpegs()
            hub.set_preview(slot, names, blobs)
            reply(
                {"ok": True, "scene": scene, "cameras": names},
                blobs=blobs,
            )
        elif op == "preview":
            slot = str(message.get("slot") or "infer")
            names, blobs = hub.preview_copy(slot)
            if not names:
                viewer = get_viewer(slot)
                if viewer.model is not None and viewer.data is not None:
                    hub.publish(slot, viewer, record=False)
                    hub.step()
                    names, blobs = hub.preview_copy(slot)
            reply({"ok": True, "cameras": names}, blobs=blobs)
        elif op == "encode":
            slot = str(message.get("slot") or "collect")
            gen = int(message.get("gen") or 0)
            packed = hub.take_or_wait(slot, gen, req_id_box["id"])
            if packed is not None:
                names, blobs = packed
                reply({"ok": True, "cameras": names}, blobs=blobs)
        elif op == "start_record":
            slot = str(message.get("slot") or "collect")
            viewer = get_viewer(slot)
            recorder.start(
                resolve_stored(message["folder"]),
                list(message.get("cameras") or viewer.camera_names),
                int(message.get("fps") or DEFAULT_COLLECT_FPS),
                int(message.get("width") or VIEW_SIZE),
                int(message.get("height") or VIEW_SIZE),
                viewer,
            )
            reply({"ok": True})
        elif op == "stop_record":
            with hub.lock:
                hub.jobs = deque(item for item in hub.jobs if not item[4])
            frames, dropped, forced = recorder.stop(save=bool(message.get("save", True)))
            reply(
                {
                    "ok": True,
                    "frames": frames,
                    "dropped": dropped,
                    "forced": forced,
                }
            )
        elif op == "render":
            slot = str(message.get("slot") or "infer")
            camera = str(message.get("camera") or "")
            names, blobs = hub.preview_copy(slot)
            frame = b""
            if camera in names:
                frame = blobs[names.index(camera)]
            reply({"ok": True}, blobs=[frame])
        elif op == "apply_ctrl":
            slot = str(message.get("slot") or "collect")
            viewer = get_viewer(slot)
            state = viewer.apply_ctrl(
                message.get("action") or [],
                hz=message.get("hz"),
            )
            record = bool(message.get("record"))
            recorded = False
            if record:
                frames = recorder.capture(viewer)
                recorded = bool(frames)
                if frames:
                    names = list(frames)
                    blobs = [
                        encode_jpeg(Image.fromarray(frames[name]))
                        for name in names
                    ]
                else:
                    names, blobs = viewer.preview_jpegs()
            else:
                names, blobs = viewer.preview_jpegs()
            hub.set_preview(slot, names, blobs)
            reply(
                {
                    "ok": True,
                    "state": state,
                    "action": np.rad2deg(viewer.data.ctrl[:ROBOT_ACTION_DIM])
                    .astype(np.float32)
                    .tolist(),
                    "recorded": recorded,
                    "gen": int(viewer.ctrl_gen),
                    "cameras": names,
                },
                blobs=blobs,
            )
        elif op == "reset":
            slot = str(message.get("slot") or "collect")
            viewer = get_viewer(slot)
            if viewer.model is None:
                try:
                    output_dir = resolve_stored(stored_path(slot_output_dir(slot)))
                except RuntimeError:
                    output_dir = None
                if output_dir is not None and (output_dir / "metadata.json").is_file():
                    viewer.load(output_dir)
            hub.clear_slot(slot)
            viewer.reset_home()
            names, blobs = viewer.preview_jpegs()
            hub.set_preview(slot, names, blobs)
            reply(
                {"ok": True, "cameras": names, "scene": viewer.public_metadata()},
                blobs=blobs,
            )
        elif op == "test_status":
            reply({"ok": True, **session.snapshot()})
        elif op == "test_start":
            if session.thread is not None and session.thread.is_alive():
                raise RuntimeError("test is already running")
            if viewers["infer"].metadata is None:
                raise RuntimeError("no scene loaded")
            session.stop.clear()
            session.set_status(running=True, phase="loading")
            session.thread = threading.Thread(
                target=run_test_thread,
                args=(message,),
                daemon=True,
            )
            session.thread.start()
            reply({"ok": True, **session.snapshot()})
        elif op == "test_stop":
            session.stop.set()
            session.set_status(running=False, phase="idle")
            reply({"ok": True, **session.snapshot()})
            threading.Thread(target=session.stop_join, daemon=True).start()
        else:
            reply({"ok": False, "error": f"unknown op {op}"})

    while True:
        timeout = 0.001 if hub.jobs or not obs_jobs.empty() else 0.05
        try:
            message = incoming.get(timeout=timeout)
        except queue.Empty:
            drain_obs_jobs()
            hub.step()
            continue
        if message is None:
            break
        batch = [message]
        while True:
            try:
                extra = incoming.get_nowait()
            except queue.Empty:
                break
            if extra is None:
                incoming.put(None)
                break
            batch.append(extra)
        if any(item.get("op") in WORKER_PRIORITY_OPS for item in batch):
            kept = []
            for item in batch:
                if item.get("op") in WORKER_DROP_WHEN_PRIORITY:
                    slot = str(item.get("slot") or "infer")
                    names, blobs = hub.preview_copy(slot)
                    reply(
                        {"ok": True, "cameras": names},
                        blobs=blobs,
                        req_id=item.get("id"),
                    )
                else:
                    kept.append(item)
            batch = kept
        batch.sort(
            key=lambda item: 0 if item.get("op") in WORKER_PRIORITY_OPS else 1
        )
        for message in batch:
            req_id_box["id"] = None
            try:
                handle(message)
            except Exception as error:
                reply({"ok": False, "error": str(error)})
        while True:
            if not incoming.empty():
                break
            drain_obs_jobs()
            processed, _recorded = hub.step()
            if not processed:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="robosim-at-home")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker()
        return
    ensure_data_dirs()
    preview = load_config()["environment"]["preview"]
    host = args.host or str(preview.get("host") or DEFAULT_HOST)
    port = args.port or int(preview.get("port") or DEFAULT_PORT)
    class StudioServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        request_queue_size = 64

    server = StudioServer((host, port), Handler)
    print(f"robosim-at-home at http://{host}:{port}", flush=True)
    threading.Thread(target=list_model_devices, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        WORKER.close()


if __name__ == "__main__":
    main()
