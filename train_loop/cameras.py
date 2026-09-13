from __future__ import annotations

import json
import logging
import os
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate

from lerobot.utils.collate import lerobot_collate_fn

# ======Settings=========
EPISODE_CAMERAS_NAME = "episode_cameras.json"
IMAGE_PREFIX = "observation.images."
FRAME_CACHE_ENABLED = True
FRAME_CACHE_GB = 8
FRAME_CACHE_ENV = "ROBOSIM_FRAME_CACHE"
FRAME_CACHE_GB_ENV = "ROBOSIM_FRAME_CACHE_GB"
PREFETCH_FRAMES = 64
STAMP_DIGITS = 5
# ======Settings=========


def _stamp_key(stamps) -> tuple[float, ...]:
    return tuple(round(float(ts), STAMP_DIGITS) for ts in stamps)


def frame_cache_limit(num_workers: int) -> tuple[bool, int]:
    raw_enabled = os.environ.get(FRAME_CACHE_ENV)
    if raw_enabled is None:
        enabled = FRAME_CACHE_ENABLED
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "no", "off"}
    raw_gb = os.environ.get(FRAME_CACHE_GB_ENV)
    gb = float(raw_gb) if raw_gb is not None else float(FRAME_CACHE_GB)
    if gb <= 0:
        enabled = False
    workers = max(1, int(num_workers))
    per_worker = max(0, int(gb * 1024**3) // workers)
    return enabled, per_worker


def episode_cameras_path(root: str | Path) -> Path:
    return Path(root) / "meta" / EPISODE_CAMERAS_NAME


def write_episode_cameras(root: str | Path, mapping: dict[int, list[str]]) -> Path:
    path = episode_cameras_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(int(index)): list(keys) for index, keys in mapping.items()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_episode_cameras(
    root: str | Path,
    camera_keys: list[str],
    num_episodes: int,
) -> dict[int, list[str]]:
    path = episode_cameras_path(root)
    if path.is_file():
        raw = json.loads(path.read_text())
        mapping = {}
        for key, value in raw.items():
            mapping[int(key)] = [str(item) for item in value]
        return mapping
    keys = list(camera_keys)
    return {index: keys for index in range(int(num_episodes))}


def _image_keys(sample: dict[str, Any]) -> set[str]:
    return {key for key in sample if key.startswith(IMAGE_PREFIX)}


def variable_camera_collate(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    shared = _image_keys(batch[0])
    for sample in batch[1:]:
        shared &= _image_keys(sample)
    if not shared:
        raise RuntimeError("batch has no shared cameras")
    trimmed = []
    for sample in batch:
        trimmed.append(
            {
                key: value
                for key, value in sample.items()
                if not key.startswith(IMAGE_PREFIX) or key in shared
            }
        )
    try:
        return lerobot_collate_fn(trimmed)
    except Exception:
        return default_collate(trimmed)


class _FilteredVideoQuery:
    """Picklable wrapper around DatasetReader._query_videos."""

    def __init__(
        self,
        original,
        present_by_episode: dict[int, list[str]],
        cache_frames: bool,
        cache_bytes_limit: int,
        camera_keys: list[str] | None = None,
    ) -> None:
        self.original = original
        self.present_by_episode = present_by_episode
        self.cache_frames = bool(cache_frames)
        self.cache_bytes_limit = int(cache_bytes_limit)
        self.camera_keys = list(camera_keys or [])
        self.cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.cache_bytes = 0
        self.hot_ep: int | None = None
        self._lock = threading.RLock()

    def _cameras(self, ep_idx: int) -> list[str]:
        allowed = self.present_by_episode.get(int(ep_idx))
        if allowed is not None:
            return list(allowed)
        return list(self.camera_keys)

    def _clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.cache_bytes = 0

    def _store(self, key: tuple, tensor: torch.Tensor) -> None:
        with self._lock:
            if key in self.cache:
                return
            item = tensor.detach()
            if item.device.type != "cpu":
                item = item.cpu()
            nbytes = int(item.nbytes) if hasattr(item, "nbytes") else item.numel() * item.element_size()
            if nbytes > self.cache_bytes_limit > 0:
                return
            while self.cache and self.cache_bytes + nbytes > self.cache_bytes_limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= (
                    int(old.nbytes) if hasattr(old, "nbytes") else old.numel() * old.element_size()
                )
            self.cache[key] = item
            self.cache_bytes += nbytes

    def _lookup(self, ep_idx: int, key: str, stamps) -> torch.Tensor | None:
        stamp_tuple = _stamp_key(stamps)
        with self._lock:
            exact = self.cache.get((ep_idx, key, stamp_tuple))
            if exact is not None:
                self.cache.move_to_end((ep_idx, key, stamp_tuple))
                return exact
            parts = []
            for ts in stamp_tuple:
                part_key = (ep_idx, key, (ts,))
                part = self.cache.get(part_key)
                if part is None:
                    return None
                self.cache.move_to_end(part_key)
                parts.append(part)
            if len(parts) == 1:
                return parts[0]
            return torch.stack(parts)

    def prefetch_chunk(self, ep_idx: int, stamps) -> None:
        if not self.cache_frames or self.cache_bytes_limit <= 0:
            return
        ep_idx = int(ep_idx)
        stamps = [float(ts) for ts in stamps]
        if not stamps:
            return
        cameras = self._cameras(ep_idx)
        if not cameras:
            return
        with self._lock:
            if self.hot_ep != ep_idx:
                self.cache.clear()
                self.cache_bytes = 0
                self.hot_ep = ep_idx
            already = (
                self._lookup(ep_idx, cameras[0], (stamps[0],)) is not None
                and self._lookup(ep_idx, cameras[0], (stamps[-1],)) is not None
            )
        if already:
            return
        loaded = self.original({camera: stamps for camera in cameras}, ep_idx)
        for camera, tensor in loaded.items():
            if not torch.is_tensor(tensor):
                continue
            if tensor.ndim == 3:
                self._store((ep_idx, camera, _stamp_key((stamps[0],))), tensor)
                continue
            count = min(int(tensor.shape[0]), len(stamps))
            for index in range(count):
                self._store((ep_idx, camera, _stamp_key((stamps[index],))), tensor[index])

    def __call__(self, query_timestamps, ep_idx):
        allowed = self.present_by_episode.get(int(ep_idx))
        if allowed is not None:
            query_timestamps = {
                key: value
                for key, value in query_timestamps.items()
                if key in allowed
            }
        if not self.cache_frames:
            return self.original(query_timestamps, ep_idx)
        found = {}
        missing = {}
        ep_idx = int(ep_idx)
        for key, stamps in query_timestamps.items():
            cached = self._lookup(ep_idx, key, stamps)
            if cached is None:
                missing[key] = stamps
            else:
                found[key] = cached
        if missing:
            loaded = self.original(missing, ep_idx)
            for key, tensor in loaded.items():
                self._store((ep_idx, key, _stamp_key(query_timestamps[key])), tensor)
                found[key] = tensor
        return found

    def __getstate__(self) -> dict:
        return {
            "original": self.original,
            "present_by_episode": self.present_by_episode,
            "cache_frames": self.cache_frames,
            "cache_bytes_limit": self.cache_bytes_limit,
            "camera_keys": self.camera_keys,
        }

    def __setstate__(self, state: dict) -> None:
        self.original = state["original"]
        self.present_by_episode = state["present_by_episode"]
        self.cache_frames = state["cache_frames"]
        self.cache_bytes_limit = state["cache_bytes_limit"]
        self.camera_keys = list(state.get("camera_keys") or [])
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.hot_ep = None
        self._lock = threading.RLock()


class VariableCameraDataset(Dataset):
    """Skip missing cameras per episode and optionally cache decoded frames."""

    def __init__(
        self,
        dataset,
        present_by_episode: dict[int, list[str]],
        cache_frames: bool = True,
        cache_bytes_limit: int = 0,
        frame_tables: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.present_by_episode = {
            int(index): list(keys) for index, keys in present_by_episode.items()
        }
        self.cache_frames = bool(cache_frames)
        self.cache_bytes_limit = int(cache_bytes_limit)
        self.frame_tables = frame_tables or {}
        self.camera_keys = list(getattr(getattr(dataset, "meta", None), "camera_keys", []) or [])
        self._install()

    def _install(self) -> None:
        reader = self.dataset.reader
        original = getattr(reader, "_original_query_videos", reader._query_videos)
        reader._original_query_videos = original
        reader._query_videos = _FilteredVideoQuery(
            original,
            self.present_by_episode,
            self.cache_frames,
            self.cache_bytes_limit,
            camera_keys=self.camera_keys,
        )

    def __getstate__(self) -> dict:
        reader = self.dataset.reader
        original = getattr(reader, "_original_query_videos", None)
        if original is not None:
            reader._query_videos = original
        state = {
            "dataset": self.dataset,
            "present_by_episode": self.present_by_episode,
            "cache_frames": self.cache_frames,
            "cache_bytes_limit": self.cache_bytes_limit,
            "frame_tables": self.frame_tables,
            "camera_keys": self.camera_keys,
        }
        if original is not None:
            self._install()
        return state

    def __setstate__(self, state: dict) -> None:
        self.dataset = state["dataset"]
        self.present_by_episode = state["present_by_episode"]
        self.cache_frames = state["cache_frames"]
        self.cache_bytes_limit = state["cache_bytes_limit"]
        self.frame_tables = state.get("frame_tables") or {}
        self.camera_keys = list(state.get("camera_keys") or [])
        self._install()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        episode = int(item["episode_index"])
        allowed = self.present_by_episode.get(episode)
        if allowed is not None:
            allowed_set = set(allowed)
            for key in list(item):
                if key.startswith(IMAGE_PREFIX) and key not in allowed_set:
                    item.pop(key)
        return item

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


class CameraBatchSampler(BatchSampler):
    """Keep each batch inside one camera-set so collate never pads extra views."""

    def __init__(
        self,
        groups: dict[frozenset[str], list[int]],
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.groups = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in groups.items()
            if len(indices) > 0
        }
        if not self.groups:
            raise ValueError("no frames for camera-grouped sampling")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._length = sum(self._group_batches(len(indices)) for indices in self.groups.values())

    def _group_batches(self, count: int) -> int:
        if self.drop_last:
            return count // self.batch_size
        return max(1, (count + self.batch_size - 1) // self.batch_size) if count else 0

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for indices in self.groups.values():
            order = rng.permutation(indices)
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                if len(chunk) == 0:
                    continue
                batches.append(chunk.tolist())
        rng.shuffle(batches)
        yield from batches


def camera_groups(
    present_by_episode: dict[int, list[str]],
    from_indices,
    to_indices,
    drop_n_last_frames: int = 0,
    absolute_to_relative_idx: dict[int, int] | None = None,
) -> dict[frozenset[str], list[int]]:
    groups: dict[frozenset[str], list[int]] = defaultdict(list)
    from_indices = np.asarray(from_indices)
    to_indices = np.asarray(to_indices)
    for episode, cameras in present_by_episode.items():
        if episode >= len(from_indices):
            continue
        start = int(from_indices[episode])
        stop = int(to_indices[episode]) - int(drop_n_last_frames)
        if stop <= start:
            continue
        key = frozenset(cameras)
        for absolute in range(start, stop):
            index = (
                absolute_to_relative_idx[absolute]
                if absolute_to_relative_idx is not None
                else absolute
            )
            groups[key].append(int(index))
    return dict(groups)


def build_episode_tables(
    dataset,
    drop_n_last_frames: int = 0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    meta = getattr(dataset, "meta", None)
    if meta is None or not hasattr(meta, "episodes"):
        return {}
    from_indices = np.asarray(meta.episodes["dataset_from_index"])
    to_indices = np.asarray(meta.episodes["dataset_to_index"])
    abs_map = getattr(dataset, "absolute_to_relative_idx", None)
    used = getattr(dataset, "episodes", None)
    if used is None:
        used = range(len(from_indices))
    ts_arr = None
    fps = float(getattr(meta, "fps", 30) or 30)
    hf = getattr(dataset, "hf_dataset", None)
    data = getattr(hf, "data", None) if hf is not None else None
    if data is not None and hasattr(data, "column"):
        try:
            ts_arr = np.asarray(data.column("timestamp").to_numpy())
        except Exception:
            ts_arr = None
    tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    drop = int(drop_n_last_frames)
    for episode in used:
        episode = int(episode)
        if episode >= len(from_indices):
            continue
        start = int(from_indices[episode])
        stop = int(to_indices[episode]) - drop
        if stop <= start:
            continue
        rels = np.empty(stop - start, dtype=np.int64)
        for offset, absolute in enumerate(range(start, stop)):
            rels[offset] = abs_map[absolute] if abs_map is not None else absolute
        if ts_arr is not None:
            stamps = np.asarray(ts_arr[rels], dtype=np.float64)
        else:
            stamps = np.arange(len(rels), dtype=np.float64) / fps
        tables[episode] = (rels, stamps)
    return tables


def _pin_batch(batch: dict[str, Any]) -> dict[str, Any]:
    for key, value in batch.items():
        if torch.is_tensor(value) and value.device.type == "cpu":
            batch[key] = value.pin_memory()
    return batch


def _yield_batch(samples: list, pin_memory: bool):
    batch = variable_camera_collate(samples)
    if batch is None:
        return None
    if pin_memory:
        batch = _pin_batch(batch)
    return batch


def iter_camera_batches(samples, batch_size: int, pin_memory: bool = False):
    buffers: dict[frozenset[str], list] = defaultdict(list)
    size = int(batch_size)
    for sample in samples:
        if sample is None:
            for key, buf in list(buffers.items()):
                if not buf:
                    continue
                batch = _yield_batch(buf, pin_memory)
                buf.clear()
                if batch is not None:
                    yield batch
            continue
        key = frozenset(_image_keys(sample))
        buf = buffers[key]
        buf.append(sample)
        if len(buf) < size:
            continue
        batch = _yield_batch(buf[:size], pin_memory)
        del buf[:size]
        if batch is not None:
            yield batch


def _prefetch_window(query, episode: int, stamps, start: int, window: int) -> None:
    if not hasattr(query, "prefetch_chunk"):
        return
    query.prefetch_chunk(episode, stamps[start : start + window])


class EpisodeStreamDataset(IterableDataset):
    """Shard episodes across workers and yield collated sequential batches."""

    def __init__(
        self,
        dataset: VariableCameraDataset,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        episodes = [int(episode) for episode in self.dataset.frame_tables]
        mine = episodes[worker_id::num_workers]
        if not mine:
            return
        rng = np.random.RandomState(self.seed + worker_id * 10007)
        query = self.dataset.dataset.reader._query_videos
        batch_size = max(1, self.batch_size)
        window = max(PREFETCH_FRAMES, batch_size)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            while True:
                for episode in rng.permutation(mine):
                    episode = int(episode)
                    indices, stamps = self.dataset.frame_tables[episode]
                    count = len(indices)
                    if count == 0:
                        continue
                    starts = list(range(0, count, window))
                    pending = pool.submit(
                        _prefetch_window, query, episode, stamps, starts[0], window
                    )
                    for offset, start in enumerate(starts):
                        pending.result()
                        nxt = offset + 1
                        if nxt < len(starts):
                            pending = pool.submit(
                                _prefetch_window,
                                query,
                                episode,
                                stamps,
                                starts[nxt],
                                window,
                            )
                        stop = min(start + window, count)
                        for pos in range(start, stop, batch_size):
                            samples = [
                                self.dataset[int(index)]
                                for index in indices[pos : pos + batch_size]
                            ]
                            batch = variable_camera_collate(samples)
                            if batch is not None:
                                yield batch
        finally:
            pool.shutdown(wait=False)


def wrap_dataset(
    dataset,
    cache_frames: bool | None = None,
    num_workers: int = 0,
    drop_n_last_frames: int = 0,
) -> VariableCameraDataset:
    enabled, per_worker = frame_cache_limit(num_workers)
    if cache_frames is None:
        cache_frames = enabled
    present = load_episode_cameras(
        dataset.root,
        list(dataset.meta.camera_keys),
        int(dataset.num_episodes),
    )
    tables = build_episode_tables(dataset, drop_n_last_frames)
    if cache_frames:
        logging.info(
            f"Frame decode cache · {per_worker / 1024**3:.2f} GB/worker · "
            f"{max(1, int(num_workers))} workers · prefetch {PREFETCH_FRAMES} frames"
        )
    else:
        logging.info("Frame decode cache off")
    return VariableCameraDataset(
        dataset,
        present,
        cache_frames=cache_frames,
        cache_bytes_limit=per_worker if cache_frames else 0,
        frame_tables=tables,
    )
