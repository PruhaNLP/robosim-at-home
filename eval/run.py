"""Batch ODE eval on a fixed validation scene set."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from core.config import EVAL_DIR, policy_cameras_from_config
from core.environment import RandomSceneEnv
from model.inference import CHUNK_SIZE, SmolVLAEngine

from .dataset import load_manifest, scene_dir

# ======Settings=========
LAST_NAME = "last.json"
# ======Settings=========


@dataclass
class EvalHooks:
    should_stop: callable
    log: callable
    progress: callable | None = None
    result: callable | None = None
    metrics: callable | None = None


def _finish(env, steps: int) -> dict:
    reward = env.finalize_reward()
    hz = float(env.control_hz)
    return {
        "success": bool(reward.success),
        "steps": int(steps),
        "seconds": float(steps) / hz if hz else 0.0,
    }


def _run_wave(
    envs: list,
    instructions: list[str],
    engine: SmolVLAEngine,
    n_action_steps: int,
    num_steps: int,
    should_stop,
    on_log=None,
) -> list[dict]:
    observations = [env.reset() for env in envs]
    steps = [0 for _ in envs]
    finished = [False for _ in envs]
    results: list[dict | None] = [None for _ in envs]
    max_steps = max(
        1,
        int(float(envs[0].rollout_config["duration_seconds"]) * envs[0].control_hz),
    )
    take = max(1, min(int(n_action_steps), CHUNK_SIZE))
    names = [
        name for name in engine.cameras if name in envs[0].camera_names
    ] or list(envs[0].camera_names)
    wave_n = len(envs)

    while (not all(finished)) and not should_stop():
        pending = [
            index
            for index in range(len(envs))
            if not finished[index] and steps[index] < max_steps
        ]
        if not pending:
            break
        items = []
        for index in pending:
            frames = {name: envs[index].render_camera(name) for name in names}
            items.append(
                (
                    frames,
                    np.asarray(observations[index]["state"], dtype=np.float32),
                    instructions[index],
                )
            )
        while len(items) < wave_n:
            items.append(items[-1])
        chunks = engine.predict_chunks(items, num_steps=int(num_steps))
        actions: list[np.ndarray | None] = [None] * len(envs)
        for row, index in enumerate(pending):
            chunk = chunks[row]
            if not np.isfinite(chunk).all():
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                if on_log is not None:
                    on_log("Eval chunk had non-finite actions, zeroed.", "error")
            actions[index] = chunk
        for tick in range(take):
            if should_stop() or all(finished):
                break
            moved = False
            for index, env in enumerate(envs):
                chunk = actions[index]
                if chunk is None or finished[index] or tick >= len(chunk):
                    continue
                if steps[index] >= max_steps:
                    results[index] = _finish(env, steps[index])
                    finished[index] = True
                    continue
                observations[index], done = env.step(
                    np.asarray(chunk[tick], dtype=np.float32),
                    render_images=False,
                )
                steps[index] += 1
                moved = True
                if done or steps[index] >= max_steps:
                    results[index] = _finish(env, steps[index])
                    finished[index] = True
            if not moved:
                break
    for index, env in enumerate(envs):
        if results[index] is None:
            results[index] = _finish(env, steps[index])
    return results


def run_eval(
    config: dict,
    checkpoint: str,
    device,
    hooks: EvalHooks,
    duration_seconds: float,
    parallel: int,
    n_action_steps: int,
    num_steps: int,
) -> dict:
    from grpo.policy import resolve_checkpoint
    from model.act import is_act_checkpoint
    from model.turbovla import TurboEngine, is_turbovla_checkpoint

    manifest = load_manifest()
    scenes = list(manifest.get("scenes") or [])
    if not scenes:
        raise RuntimeError("build a validation set first")
    resolved = resolve_checkpoint(checkpoint)
    if is_act_checkpoint(resolved):
        raise RuntimeError("Eval batch does not support ACT checkpoints.")
    cameras = policy_cameras_from_config(config)
    if is_turbovla_checkpoint(resolved):
        engine = TurboEngine(device=str(device))
        engine.load(resolved, on_log=hooks.log)
        if not engine.has_stats():
            raise RuntimeError("TurboVLA base has no SO-100 stats yet; fine-tune it first")
        engine.map_cameras(cameras)
    else:
        engine = SmolVLAEngine(device=str(device))
        engine.load(source=resolved, on_log=hooks.log, cameras=cameras or None, warmup=False)
    runtime = dict(config)
    runtime["environment"] = dict(config["environment"])
    runtime["environment"]["rollout"] = dict(config["environment"]["rollout"])
    runtime["environment"]["rollout"]["duration_seconds"] = max(1.0, float(duration_seconds))
    parallel = max(1, min(int(parallel), len(scenes)))
    hooks.log(
        f"Eval · {len(scenes)} scenes · together {parallel} · "
        f"{float(runtime['environment']['rollout']['duration_seconds']):g}s · "
        f"denoise {int(num_steps)}."
    )
    results: list[dict] = []
    n_success = 0
    offset = 0
    while offset < len(scenes) and not hooks.should_stop():
        wave = scenes[offset : offset + parallel]
        if hooks.progress is not None:
            hooks.progress(
                "eval",
                offset,
                len(scenes),
                f"Scene {offset + 1}/{len(scenes)}",
            )
        envs = []
        try:
            for item in wave:
                dest = scene_dir(int(item["index"]))
                if not (dest / "metadata.json").is_file():
                    raise RuntimeError(f"val scene {item['index']} is missing")
                envs.append(
                    RandomSceneEnv(dest, runtime, sensor_seed=int(item["seed"]))
                )
            wave_out = _run_wave(
                envs,
                [str(item.get("instruction") or "") for item in wave],
                engine,
                n_action_steps,
                num_steps,
                hooks.should_stop,
                hooks.log,
            )
        finally:
            for env in envs:
                env.close()
        for item, out in zip(wave, wave_out, strict=True):
            row = {
                "index": int(item["index"]),
                "seed": int(item["seed"]),
                "instruction": str(item.get("instruction") or ""),
                "target": str(item.get("target") or ""),
                **out,
            }
            results.append(row)
            if row["success"]:
                n_success += 1
            if hooks.result is not None:
                hooks.result(row)
            kind = "ok" if row["success"] else "info"
            hooks.log(
                f"Scene {row['index'] + 1}/{len(scenes)} · "
                f"{'success' if row['success'] else 'fail'} · "
                f"{row['seconds']:.1f}s.",
                kind,
            )
        offset += len(wave)
        if hooks.metrics is not None:
            hooks.metrics(
                {
                    "n_scenes": len(results),
                    "n_success": n_success,
                    "success_rate": n_success / max(1, len(results)),
                    "checkpoint": str(resolved),
                }
            )
    if hooks.should_stop() and len(results) < len(scenes):
        hooks.log("Eval stopped.", "warn")
    summary = {
        "n_scenes": len(results),
        "n_success": n_success,
        "success_rate": n_success / max(1, len(results)),
        "checkpoint": str(resolved),
        "results": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / LAST_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    if hooks.progress is not None:
        hooks.progress("done", len(results), len(scenes), "Done")
    hooks.log(
        f"Done. {n_success}/{len(results)} success "
        f"({summary['success_rate']:.2f}).",
        "ok",
    )
    return summary
