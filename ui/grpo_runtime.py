from __future__ import annotations

import threading
import time

from core.config import (
    GRPO_SCENE_DIR,
    LLM_LOG_PATH,
    TRAIN_DIR,
    normalize_policy_mode,
    stored_path,
)
from devices import apply_render_device, resolve_model_device, resolve_render_device
from train import _slug, list_checkpoints

# ======Settings=========
DEFAULT_RUN = "grpo"
LOG_LIMIT = 400
HISTORY_LIMIT = 240
HISTORY_KEYS = ("reward", "success", "loss", "clip", "drift", "kl", "gradNorm")
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


class GrpoRuntime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.running = False
        self.error: str | None = None
        self.logs: list[dict] = []
        self.checkpoint = ""
        self.run = DEFAULT_RUN
        self.output_dir: str | None = None
        self.progress = _idle_progress()
        self.history: dict[str, list[dict]] = {key: [] for key in HISTORY_KEYS}
        self.scene: dict | None = None
        self.group: list[dict] = []
        self.metrics: dict | None = None
        self._progress_t0 = time.monotonic()

    def snapshot(self, extras: bool = True) -> dict:
        with self.lock:
            payload = {
                "running": self.running,
                "checkpoint": self.checkpoint,
                "run": self.run,
                "outputDir": self.output_dir,
                "error": self.error,
                "logs": list(self.logs[-LOG_LIMIT:]),
                "progress": dict(self.progress),
                "history": {key: list(points) for key, points in self.history.items()},
                "scene": dict(self.scene) if self.scene else None,
                "group": list(self.group),
                "metrics": dict(self.metrics) if self.metrics else None,
            }
        if extras:
            payload["checkpoints"] = list_checkpoints()
        return payload

    def log(self, text: str, kind: str = "info") -> None:
        line = str(text or "").strip()
        if not line:
            return
        with self.lock:
            self.logs.append({"kind": kind, "text": line})
            self.logs = self.logs[-LOG_LIMIT:]

    def start(self, payload: dict, compute: dict, config: dict, preview, worker=None) -> dict:
        mode = normalize_policy_mode(payload.get("policy_mode") or config.get("policy_mode"))
        if mode != "smolvla":
            raise RuntimeError(f"GRPO is not available in {mode} mode.")
        with self.lock:
            if self.running:
                raise RuntimeError("GRPO is already running")
            self.checkpoint = str(payload.get("checkpoint") or "").strip()
            if not self.checkpoint:
                raise RuntimeError("select a checkpoint")
            self.run = _slug(payload.get("run") or DEFAULT_RUN)
            self.error = None
            self.output_dir = None
            self.logs = []
            self.progress = _idle_progress()
            self.history = {key: [] for key in HISTORY_KEYS}
            self.scene = None
            self.group = []
            self.metrics = None
            self.stop.clear()
            self.running = True
            self._progress_t0 = time.monotonic()
        self.log(f"Prepare GRPO run {self.run}.")
        self.thread = threading.Thread(
            target=self._run,
            args=(dict(payload), compute, config, preview, worker),
            daemon=True,
        )
        self.thread.start()
        return self.snapshot()

    def stop_run(self) -> dict:
        self.stop.set()
        self.log("Stopping…", "warn")
        return self.snapshot()

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

    def _apply_metrics(self, info: dict) -> None:
        step = int(info["update"])
        mapping = {
            "reward": info.get("mean_reward"),
            "success": info.get("success_rate"),
            "loss": info.get("loss"),
            "clip": info.get("clip_fraction"),
            "drift": info.get("drift"),
            "kl": info.get("kl"),
            "gradNorm": info.get("grad_norm"),
        }
        with self.lock:
            self.metrics = dict(info)
            for key, value in mapping.items():
                if value is None:
                    continue
                self.history[key].append({"step": step, "value": float(value)})
                self.history[key] = self.history[key][-HISTORY_LIMIT:]
            total = int(self.progress.get("total") or 0)
            self.progress.update(
                {
                    "phase": "update",
                    "current": step,
                    "percent": min(100.0, round(100.0 * step / total, 1)) if total else 0.0,
                    "label": f"Update {step}" + (f"/{total}" if total else ""),
                }
            )

    def _run(self, payload: dict, compute: dict, config: dict, preview, worker=None) -> None:
        try:
            output = TRAIN_DIR / self.run
            if output.exists():
                raise RuntimeError(f"run exists: {output}")
            if worker is not None:
                worker.close()
            render = resolve_render_device(compute)
            apply_render_device(render)
            device_name = resolve_model_device(compute.get("model"))
            import torch

            device = torch.device(device_name)
            from grpo.train import GrpoHooks, run_grpo

            def should_stop() -> bool:
                return self.stop.is_set()

            def on_preview(cameras: list[str], blobs: list[bytes]) -> None:
                preview.put("grpo", cameras, blobs)

            def on_scene(scene: dict) -> None:
                with self.lock:
                    self.scene = scene
                preview.set_scene("grpo", scene)

            def on_group(members: list[dict]) -> None:
                with self.lock:
                    self.group = members

            with self.lock:
                self.output_dir = stored_path(output)
            runtime = dict(config)
            runtime["compute"] = dict(config.get("compute") or {})
            runtime["policy_mode"] = str(
                payload.get("policy_mode") or config.get("policy_mode") or "smolvla"
            )
            runtime["environment"] = dict(config["environment"])
            runtime["environment"]["rollout"] = dict(config["environment"]["rollout"])
            runtime["environment"]["grpo"] = dict(config["environment"]["grpo"])
            grpo = runtime["environment"]["grpo"]
            flow = dict(grpo.get("flow") or {})
            opt = dict(grpo.get("optimization") or {})
            if payload.get("duration_seconds") is not None:
                runtime["environment"]["rollout"]["duration_seconds"] = max(
                    1.0, float(payload["duration_seconds"])
                )
            if payload.get("parallel_rollouts") is not None:
                grpo["parallel_rollouts"] = max(1, int(payload["parallel_rollouts"]))
            if payload.get("group_size") is not None:
                grpo["group_size"] = max(2, int(payload["group_size"]))
            if payload.get("scenes_per_update") is not None:
                grpo["scenes_per_update"] = max(1, int(payload["scenes_per_update"]))
            if payload.get("scene_waves") is not None:
                grpo["scene_waves"] = max(1, int(payload["scene_waves"]))
            if payload.get("max_updates") is not None:
                grpo["max_updates"] = payload["max_updates"]
            if payload.get("train_scope"):
                grpo["train_scope"] = str(payload["train_scope"])
            if payload.get("sde_mode"):
                flow["sde_mode"] = str(payload["sde_mode"])
            if payload.get("noise_level") is not None:
                flow["noise_level"] = float(payload["noise_level"])
            if payload.get("denoising_steps") is not None:
                flow["denoising_steps"] = max(1, int(payload["denoising_steps"]))
            if payload.get("expert_lr") is not None:
                opt["action_expert_learning_rate"] = float(payload["expert_lr"])
            if payload.get("vlm_lr") is not None:
                opt["vlm_learning_rate"] = float(payload["vlm_lr"])
            if payload.get("clip_eps") is not None:
                grpo["clip_eps"] = float(payload["clip_eps"])
            if payload.get("kl_coef") is not None:
                grpo["kl_coef"] = float(payload["kl_coef"])
            if payload.get("n_action_steps") is not None:
                grpo["n_action_steps"] = max(1, int(payload["n_action_steps"]))
            grpo["parallel_rollouts"] = max(
                1, min(int(grpo.get("parallel_rollouts") or 1), int(grpo["group_size"]))
            )
            grpo["scene_waves"] = max(
                1,
                min(
                    int(grpo.get("scene_waves") or 1),
                    int(grpo["scenes_per_update"]),
                ),
            )
            grpo["flow"] = flow
            grpo["optimization"] = opt
            GRPO_SCENE_DIR.mkdir(parents=True, exist_ok=True)
            self._set_progress("prepare", 0, 1, f"Load {self.checkpoint}")
            run_grpo(
                runtime,
                self.checkpoint,
                output,
                device,
                GrpoHooks(
                    should_stop=should_stop,
                    log=self.log,
                    preview=on_preview,
                    metrics=self._apply_metrics,
                    group=on_group,
                    progress=self._set_progress,
                    scene=on_scene,
                ),
                GRPO_SCENE_DIR,
                LLM_LOG_PATH,
            )
            self._set_progress("done", 1, 1, "Done")
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
