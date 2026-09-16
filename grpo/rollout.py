"""Group rollouts in RandomSceneEnv with Prism rewards."""
from __future__ import annotations

import io
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rewards.prism import RawRolloutReward, score_group

from .policy import FlowSdePolicy, SdeRollout

# ======Settings=========
JPEG_QUALITY = 85
LLM_SOURCE = "grpo"
DEFAULT_PREVIEW_HZ = 15
STREAM_QUEUE_MAX = 512
STEP_WORKERS = 32
# ======Settings=========


class _BgWork:
    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._done = threading.Event()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job()
            except BaseException as exc:
                self._exc = exc
            self._done.set()

    def run(self, job) -> None:
        self._exc = None
        self._done.clear()
        self._jobs.put(job)

    def wait(self) -> None:
        self._done.wait()
        if self._exc is not None:
            raise self._exc

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5.0)


def _split_waves(count: int, wave_size: int) -> list[list[int]]:
    size = max(1, min(int(wave_size), count))
    return [
        list(range(start, min(start + size, count)))
        for start in range(0, count, size)
    ]


@dataclass
class ChunkRecord:
    obs: dict
    sde: SdeRollout


@dataclass
class EpisodeRecord:
    member: int
    task: str
    cameras: list[str]
    success: bool
    n_env_steps: int
    n_chunks: int
    raw: RawRolloutReward
    scored: dict
    advantage: float = 0.0
    chunks: list[ChunkRecord] = field(default_factory=list)


class _StreamPump:
    """Play live qpos at stream FPS on a side thread so infer does not freeze the view."""

    def __init__(self, env, on_preview, fps: float) -> None:
        self._on_preview = on_preview
        self._env = env
        self._fps = max(1.0, float(fps) if fps else DEFAULT_PREVIEW_HZ)
        self._model = env.model
        self._height = int(env.renderer.height)
        self._width = int(env.renderer.width)
        self._cameras = list(env.camera_names)
        self._nq = int(env.data.qpos.shape[0])
        self._nv = int(env.data.qvel.shape[0])
        self._q: queue.Queue = queue.Queue(maxsize=STREAM_QUEUE_MAX)
        self._stop = threading.Event()
        self._broken = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, data) -> None:
        if self._on_preview is None or self._stop.is_set():
            return
        if self._broken:
            self._on_preview(
                {name: self._env.render_camera_raw(name) for name in self._cameras}
            )
            return
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        if qpos.shape[0] != self._nq or qvel.shape[0] != self._nv:
            return
        item = (np.array(qpos, copy=True), np.array(qvel, copy=True))
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=5.0)
        self._env = None
        self._model = None
        self._on_preview = None

    def _run(self) -> None:
        import mujoco
        from cameras.cameras import SharedRenderer

        try:
            renderer = SharedRenderer(self._model, self._height, self._width)
        except Exception:
            self._broken = True
            return
        data = mujoco.MjData(self._model)
        interval = 1.0 / self._fps
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                qpos, qvel = item
                if qpos.shape[0] != data.qpos.shape[0] or qvel.shape[0] != data.qvel.shape[0]:
                    continue
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(self._model, data)
                frames = {}
                with renderer.lock:
                    for name in self._cameras:
                        frames[name] = Image.fromarray(
                            renderer.render_rgb(data, name, copy=True)
                        )
                self._on_preview(frames)
                next_tick += interval
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.perf_counter()
        finally:
            renderer.close()


def encode_jpeg(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _policy_cameras(env, actor: FlowSdePolicy) -> list[str]:
    wanted = [
        name
        for name in (getattr(actor.engine, "cameras", None) or [])
        if name in env.camera_names
    ]
    return wanted or list(env.camera_names)


def _normal(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().cpu()


def _cpu_obs(obs: dict) -> dict:
    out = {}
    for key, value in obs.items():
        out[key] = _normal(value) if torch.is_tensor(value) else value
    return out


def _cpu_sde(rollout: SdeRollout) -> SdeRollout:
    return SdeRollout(
        action=_normal(rollout.action),
        traj=_normal(rollout.traj),
        taus=_normal(rollout.taus),
        dtau=rollout.dtau,
        step_logprobs=_normal(rollout.step_logprobs),
        step_mask=_normal(rollout.step_mask),
        step_weights=_normal(rollout.step_weights),
    )


def _log_llm(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _infer_chunk(
    env,
    actor: FlowSdePolicy,
    observation: dict,
    instruction: str,
    llm_log_path: Path,
    on_log=None,
) -> tuple[ChunkRecord, np.ndarray]:
    names = _policy_cameras(env, actor)
    frames = {name: env.render_camera(name) for name in names}
    raw = actor.engine.prepare_obs(
        frames,
        np.asarray(observation["state"], dtype=np.float32),
        instruction,
    )
    started = time.perf_counter()
    event = {
        "timestamp": time.time(),
        "source": LLM_SOURCE,
        "device": str(actor.engine.device),
        "instruction": instruction,
        "cameras": names,
        "num_steps": getattr(actor, "num_steps", 1),
        "sde_mode": getattr(actor, "sde_mode", "act"),
    }
    try:
        sde = actor.rollout(raw)
        event["ok"] = True
    except Exception as error:
        event["ok"] = False
        event["error"] = str(error)
        raise
    finally:
        event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        _log_llm(llm_log_path, event)
    chunk = actor.engine.actions_to_env(sde.action)
    if not np.isfinite(chunk).all():
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        if on_log is not None:
            on_log("SDE chunk had non-finite actions, zeroed.", "error")
    return ChunkRecord(obs=_cpu_obs(raw), sde=_cpu_sde(sde)), chunk


def _stack_obs(obs_list: list[dict]) -> dict:
    def _row(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.dim() >= 2 else tensor.unsqueeze(0)

    return {
        "images": torch.stack([item["images"] for item in obs_list], dim=0),
        "n_real": int(obs_list[0]["n_real"]),
        "tokens": torch.cat([_row(item["tokens"]) for item in obs_list], dim=0),
        "mask": torch.cat([_row(item["mask"]) for item in obs_list], dim=0),
        "state": torch.cat([item["state"] for item in obs_list], dim=0),
    }


def _instruction_at(instruction: str | list[str], index: int) -> str:
    if isinstance(instruction, (list, tuple)):
        return str(instruction[index])
    return str(instruction)


def _slice_sde(sde: SdeRollout, row: int) -> SdeRollout:
    logp = sde.step_logprobs
    if logp.ndim == 2:
        logp = logp[:, row : row + 1]
    return SdeRollout(
        action=sde.action[row : row + 1],
        traj=sde.traj[:, row : row + 1],
        taus=sde.taus,
        dtau=sde.dtau,
        step_logprobs=logp,
        step_mask=sde.step_mask,
        step_weights=sde.step_weights,
    )


def _render_frames(
    envs: list,
    pending: list[int],
    names: list[str],
    pool: ThreadPoolExecutor | None,
) -> dict[int, dict]:
    groups: dict[int, list[int]] = {}
    for index in pending:
        groups.setdefault(id(envs[index].renderer), []).append(index)

    def _one(indices: list[int]) -> dict[int, dict]:
        out = {}
        for index in indices:
            out[index] = {name: envs[index].render_camera(name) for name in names}
        return out

    items = list(groups.values())
    if not items:
        return {}
    if pool is None or len(items) <= 1:
        return _one(items[0])
    frames: dict[int, dict] = {}
    for part in pool.map(_one, items):
        frames.update(part)
    return frames


def _prepare_raws(
    actor: FlowSdePolicy,
    observations: list[dict],
    instruction: str | list[str],
    pending: list[int],
    frames_map: dict[int, dict],
) -> tuple[list[str], list[dict]]:
    texts = [_instruction_at(instruction, index) for index in pending]
    raws = [
        actor.engine.prepare_obs(
            frames_map[index],
            np.asarray(observations[index]["state"], dtype=np.float32),
            text,
        )
        for index, text in zip(pending, texts, strict=True)
    ]
    return texts, raws


def _run_sde(
    actor: FlowSdePolicy,
    raws: list[dict],
    pending: list[int],
    texts: list[str],
    names: list[str],
    llm_log_path: Path,
    on_log=None,
) -> dict[int, tuple[ChunkRecord, np.ndarray]]:
    started = time.perf_counter()
    event = {
        "timestamp": time.time(),
        "source": LLM_SOURCE,
        "device": str(actor.engine.device),
        "instruction": texts[0] if len(set(texts)) == 1 else texts,
        "cameras": names,
        "num_steps": getattr(actor, "num_steps", 1),
        "sde_mode": getattr(actor, "sde_mode", "act"),
        "batch": len(pending),
    }
    try:
        packed = raws[0] if len(raws) == 1 else _stack_obs(raws)
        sde = actor.rollout(packed)
        event["ok"] = True
    except Exception as error:
        event["ok"] = False
        event["error"] = str(error)
        raise
    finally:
        event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        _log_llm(llm_log_path, event)
    out: dict[int, tuple[ChunkRecord, np.ndarray]] = {}
    for row, index in enumerate(pending):
        row_sde = _slice_sde(sde, row)
        chunk = actor.engine.actions_to_env(row_sde.action, 0)
        if not np.isfinite(chunk).all():
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            if on_log is not None:
                on_log("SDE chunk had non-finite actions, zeroed.", "error")
        out[index] = (ChunkRecord(obs=_cpu_obs(raws[row]), sde=_cpu_sde(row_sde)), chunk)
    return out


def _infer_chunks(
    envs: list,
    actor: FlowSdePolicy,
    observations: list[dict],
    instruction: str | list[str],
    llm_log_path: Path,
    pending: list[int],
    on_log=None,
    pool: ThreadPoolExecutor | None = None,
    prepared: tuple[list[str], list[dict]] | None = None,
) -> dict[int, tuple[ChunkRecord, np.ndarray]]:
    names = _policy_cameras(envs[pending[0]], actor)
    if prepared is None:
        frames_map = _render_frames(envs, pending, names, pool)
        texts, raws = _prepare_raws(actor, observations, instruction, pending, frames_map)
    else:
        texts, raws = prepared
    return _run_sde(actor, raws, pending, texts, names, llm_log_path, on_log)


def _finish_episode(
    env,
    instruction: str,
    steps: int,
    chunks: list[ChunkRecord],
) -> EpisodeRecord:
    raw_reward = env.finalize_reward()
    return EpisodeRecord(
        member=0,
        task=instruction,
        cameras=list(env.camera_names),
        success=bool(raw_reward.success),
        n_env_steps=steps,
        n_chunks=len(chunks),
        raw=raw_reward,
        scored={},
        chunks=chunks,
    )


def run_episode(
    env,
    actor: FlowSdePolicy,
    instruction: str,
    n_action_steps: int,
    llm_log_path: Path,
    should_stop,
    on_preview=None,
    on_log=None,
    preview_hz: float = DEFAULT_PREVIEW_HZ,
    pump: _StreamPump | None = None,
) -> EpisodeRecord:
    observation = env.reset()
    chunks: list[ChunkRecord] = []
    steps = 0
    max_steps = max(1, int(float(env.rollout_config["duration_seconds"]) * env.control_hz))
    from model.inference import CHUNK_SIZE

    take = max(1, min(int(n_action_steps), CHUNK_SIZE))
    own_pump = pump is None and on_preview is not None
    if own_pump:
        pump = _StreamPump(env, on_preview, preview_hz)
    try:
        if pump is not None:
            pump.push(env.data)
        while steps < max_steps and not should_stop():
            record, chunk = _infer_chunk(
                env, actor, observation, instruction, llm_log_path, on_log
            )
            chunks.append(record)
            for action in chunk[:take]:
                if steps >= max_steps or should_stop():
                    break
                observation, done = env.step(
                    np.asarray(action, dtype=np.float32), render_images=False
                )
                steps += 1
                if pump is not None:
                    pump.push(env.data)
                if done:
                    return _finish_episode(env, instruction, steps, chunks)
        return _finish_episode(env, instruction, steps, chunks)
    finally:
        if own_pump and pump is not None:
            pump.close()


def run_episode_batch(
    envs: list,
    actor: FlowSdePolicy,
    instruction: str | list[str],
    n_action_steps: int,
    llm_log_path: Path,
    should_stop,
    on_preview=None,
    on_log=None,
    stream_at: int | None = None,
    preview_hz: float = DEFAULT_PREVIEW_HZ,
    pump: _StreamPump | None = None,
    wave_size: int | None = None,
) -> list[EpisodeRecord]:
    n_workers = min(STEP_WORKERS, max(1, len(envs)))
    cpu_pool = ThreadPoolExecutor(max_workers=n_workers)
    if len(envs) == 1:
        observations = [envs[0].reset()]
    else:
        observations = list(cpu_pool.map(lambda env: env.reset(), envs))
    chunk_lists: list[list[ChunkRecord]] = [[] for _ in envs]
    steps = [0 for _ in envs]
    finished = [False for _ in envs]
    results: list[EpisodeRecord | None] = [None for _ in envs]
    max_steps = max(
        1,
        int(float(envs[0].rollout_config["duration_seconds"]) * envs[0].control_hz),
    )
    from model.inference import CHUNK_SIZE

    take = max(1, min(int(n_action_steps), CHUNK_SIZE))

    preferred = 0 if stream_at is None else max(0, min(int(stream_at), len(envs) - 1))

    def live_index() -> int:
        if not finished[preferred]:
            return preferred
        for index, done in enumerate(finished):
            if not done:
                return index
        return 0

    own_pump = pump is None and on_preview is not None
    if own_pump:
        pump = _StreamPump(envs[preferred], on_preview, preview_hz)
    waves = _split_waves(len(envs), wave_size or len(envs))
    actions: list[np.ndarray | None] = [None] * len(envs)
    worker = _BgWork() if len(waves) > 1 else None
    prefetch: dict[tuple[int, ...], tuple[list[int], dict[int, dict]]] = {}

    def _needs_infer(wave: list[int]) -> bool:
        return any(
            (not finished[index]) and steps[index] < max_steps and actions[index] is None
            for index in wave
        )

    def _has_actions(wave: list[int]) -> bool:
        return any(
            actions[index] is not None and not finished[index] for index in wave
        )

    def _pending(wave: list[int]) -> list[int]:
        return [
            index
            for index in wave
            if not finished[index] and steps[index] < max_steps and actions[index] is None
        ]

    def _apply_inferred(
        pending: list[int],
        inferred: dict[int, tuple[ChunkRecord, np.ndarray]],
    ) -> None:
        for index, (record, chunk) in inferred.items():
            chunk_lists[index].append(record)
            actions[index] = chunk
        for index in pending:
            if actions[index] is None and not finished[index]:
                results[index] = _finish_episode(
                    envs[index],
                    _instruction_at(instruction, index),
                    steps[index],
                    chunk_lists[index],
                )
                finished[index] = True

    def _prefetch_render(wave: list[int]) -> tuple[list[int], dict[int, dict]] | None:
        pending = _pending(wave)
        if not pending:
            return None
        names = _policy_cameras(envs[pending[0]], actor)
        return pending, _render_frames(envs, pending, names, cpu_pool)

    def _infer_wave(wave: list[int]) -> None:
        key = tuple(wave)
        prepared = prefetch.pop(key, None)
        pending = _pending(wave) if prepared is None else [
            index
            for index in prepared[0]
            if not finished[index] and steps[index] < max_steps and actions[index] is None
        ]
        if not pending:
            return
        names = _policy_cameras(envs[pending[0]], actor)
        if prepared is None:
            frames_map = _render_frames(envs, pending, names, cpu_pool)
        else:
            frames_map = prepared[1]
        texts, raws = _prepare_raws(
            actor, observations, instruction, pending, frames_map
        )
        inferred = _run_sde(
            actor, raws, pending, texts, names, llm_log_path, on_log
        )
        _apply_inferred(pending, inferred)

    def _step_indices(indices: list[int]) -> None:
        jobs = [
            index
            for index in indices
            if actions[index] is not None and not finished[index]
        ]
        if not jobs:
            return
        live = live_index() if pump is not None else None

        def _one(index: int):
            chunk = actions[index]
            if chunk is None:
                return index, steps[index], False, False
            env = envs[index]
            used = min(take, len(chunk))
            new_steps, done = env.step_actions(
                chunk[:used],
                max_steps,
                steps[index],
                on_step=pump.push if pump is not None and index == live else None,
            )
            return index, new_steps, done, True

        if len(jobs) == 1:
            rows = [_one(jobs[0])]
        else:
            rows = list(cpu_pool.map(_one, jobs))
        for index, new_steps, done, moved in rows:
            if not moved:
                actions[index] = None
                continue
            steps[index] = new_steps
            observations[index] = envs[index].observation(render_images=False)
            actions[index] = None
            if done or steps[index] >= max_steps:
                results[index] = _finish_episode(
                    envs[index],
                    _instruction_at(instruction, index),
                    steps[index],
                    chunk_lists[index],
                )
                finished[index] = True

    def _step_waves(step_waves: list[list[int]]) -> None:
        indices = [index for wave in step_waves for index in wave]
        _step_indices(indices)
        for wave in step_waves:
            key = tuple(wave)
            if key in prefetch or not _needs_infer(wave):
                continue
            prepared = _prefetch_render(wave)
            if prepared is not None:
                prefetch[key] = prepared

    try:
        if pump is not None:
            pump.push(envs[live_index()].data)
        while (not all(finished)) and not should_stop():
            ready = [wave for wave in waves if _has_actions(wave)]
            need = [wave for wave in waves if _needs_infer(wave)]
            if not ready:
                if not need:
                    break
                _infer_wave(need[0])
                continue
            infer_wave = next((wave for wave in need if wave not in ready), None)
            if worker is not None and infer_wave is not None:
                worker.run(lambda waves_to_step=ready: _step_waves(waves_to_step))
                _infer_wave(infer_wave)
                worker.wait()
            else:
                _step_waves(ready)
                if infer_wave is not None:
                    _infer_wave(infer_wave)
            if pump is not None:
                pump.push(envs[live_index()].data)
        for index, env in enumerate(envs):
            if results[index] is None:
                results[index] = _finish_episode(
                    env,
                    _instruction_at(instruction, index),
                    steps[index],
                    chunk_lists[index],
                )
        return results
    finally:
        prefetch.clear()
        cpu_pool.shutdown(wait=False)
        if worker is not None:
            worker.close()
        if own_pump and pump is not None:
            pump.close()


def collect_group(
    env,
    actor: FlowSdePolicy,
    reward_config: dict,
    n_action_steps: int,
    group_size: int,
    llm_log_path: Path,
    should_stop,
    on_preview=None,
    on_log=None,
    on_member=None,
    parallel: int = 1,
    envs: list | None = None,
    preview_hz: float = DEFAULT_PREVIEW_HZ,
    wave_size: int | None = None,
) -> list[EpisodeRecord]:
    pool = list(envs) if envs else [env]
    instruction = str(pool[0].scene_metadata["instruction"])
    group_size = int(group_size)
    parallel = max(1, min(int(parallel), group_size, len(pool)))
    stream_member = int(np.random.randint(0, group_size))
    if on_preview is not None and on_log is not None:
        on_log(f"Live stream · member {stream_member + 1}/{group_size}.")
    episodes: list[EpisodeRecord] = []
    pump = (
        _StreamPump(pool[0], on_preview, preview_hz) if on_preview is not None else None
    )
    try:
        if parallel <= 1:
            worker = pool[0]
            for member in range(group_size):
                if should_stop():
                    break
                if on_log is not None:
                    on_log(f"Group member {member + 1}/{group_size}.")
                if pump is not None:
                    pump.clear()
                episode = run_episode(
                    worker,
                    actor,
                    instruction,
                    n_action_steps,
                    llm_log_path,
                    should_stop,
                    on_preview=None,
                    on_log=on_log,
                    preview_hz=preview_hz,
                    pump=pump if member == stream_member else None,
                )
                episode.member = member
                episodes.append(episode)
                if on_member is not None:
                    on_member(episode, member, group_size)
        else:
            for batch_start in range(0, group_size, parallel):
                if should_stop():
                    break
                batch = pool[: min(parallel, group_size - batch_start)]
                last = batch_start + len(batch)
                if on_log is not None:
                    on_log(f"Group members {batch_start + 1}–{last}/{group_size} together.")
                in_batch = batch_start <= stream_member < batch_start + len(batch)
                if pump is not None and in_batch:
                    pump.clear()
                batch_episodes = run_episode_batch(
                    batch,
                    actor,
                    instruction,
                    n_action_steps,
                    llm_log_path,
                    should_stop,
                    on_preview=None,
                    on_log=on_log,
                    stream_at=stream_member - batch_start if in_batch else 0,
                    preview_hz=preview_hz,
                    pump=pump if in_batch else None,
                    wave_size=wave_size or parallel,
                )
                for offset, episode in enumerate(batch_episodes):
                    episode.member = batch_start + offset
                    episodes.append(episode)
                    if on_member is not None:
                        on_member(episode, episode.member, group_size)
    finally:
        if pump is not None:
            pump.close()
    if not episodes:
        return []
    scored = score_group([item.raw for item in episodes], reward_config)
    for episode, payload in zip(episodes, scored, strict=True):
        episode.scored = payload
    return episodes


def collect_scene_wave(
    packs: list[tuple[list, str]],
    actor: FlowSdePolicy,
    reward_config: dict,
    n_action_steps: int,
    group_size: int,
    llm_log_path: Path,
    should_stop,
    on_preview=None,
    on_log=None,
    wave_size: int = 1,
    preview_hz: float = DEFAULT_PREVIEW_HZ,
) -> list[list[EpisodeRecord]]:
    if not packs:
        return []
    group_size = int(group_size)
    pool_n = min(len(pack[0]) for pack in packs)
    if pool_n < 1:
        raise RuntimeError("empty scene wave")
    wave_size = max(1, int(wave_size))
    groups: list[list[EpisodeRecord]] = [[] for _ in packs]
    stream_pack = int(np.random.randint(0, len(packs)))
    pump = (
        _StreamPump(packs[stream_pack][0][0], on_preview, preview_hz)
        if on_preview is not None
        else None
    )
    try:
        if on_preview is not None and on_log is not None:
            on_log(f"Live stream · scene {stream_pack + 1}/{len(packs)}.")
        for batch_start in range(0, group_size, pool_n):
            if should_stop():
                break
            take = min(pool_n, group_size - batch_start)
            last = batch_start + take
            if on_log is not None:
                on_log(
                    f"{len(packs)} scenes · members {batch_start + 1}–{last}/{group_size}."
                )
            flat_envs = []
            texts: list[str] = []
            for envs, instruction in packs:
                flat_envs.extend(envs[:take])
                texts.extend([instruction] * take)
            if pump is not None:
                pump.clear()
            stream_at = stream_pack * take
            batch_episodes = run_episode_batch(
                flat_envs,
                actor,
                texts,
                n_action_steps,
                llm_log_path,
                should_stop,
                on_preview=None,
                on_log=on_log,
                stream_at=stream_at,
                preview_hz=preview_hz,
                pump=pump,
                wave_size=wave_size,
            )
            for index, episode in enumerate(batch_episodes):
                pack_index = index // take
                offset = index % take
                episode.member = batch_start + offset
                groups[pack_index].append(episode)
    finally:
        if pump is not None:
            pump.close()
    out = []
    for episodes in groups:
        if not episodes:
            out.append([])
            continue
        scored = score_group([item.raw for item in episodes], reward_config)
        for episode, payload in zip(episodes, scored, strict=True):
            episode.scored = payload
        out.append(episodes)
    return out
