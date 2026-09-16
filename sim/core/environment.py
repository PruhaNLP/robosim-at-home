import gc
import json
import os
from pathlib import Path

# ======Settings=========
DEFAULT_EGL_DEVICE_ID = "0"
COLLISION_GROUP = 3
ROBOT_ACTION_DIM = 6
# ======Settings=========

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MUJOCO_EGL_DEVICE_ID",
    os.environ.get("ROBOSIM_RENDER_EGL_DEVICE_ID", DEFAULT_EGL_DEVICE_ID),
)

import mujoco
import numpy as np
from PIL import Image

from cameras.cameras import SharedRenderer, ThreadedRenderer, apply_camera_pipeline
from rewards.prism import PrismRewardTracker, RawRolloutReward


def load_scene_model(scene_dir: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(Path(scene_dir) / "scene.xml"))


def open_scene_envs(
    scene_dir: Path,
    config: dict,
    sensor_seeds: list[int],
    model: mujoco.MjModel | None = None,
    renderer: SharedRenderer | None = None,
    n_renderers: int = 1,
) -> tuple[mujoco.MjModel, list[SharedRenderer], list["RandomSceneEnv"]]:
    scene_dir = Path(scene_dir).resolve()
    model = model or load_scene_model(scene_dir)
    rollout = config["environment"]["rollout"]
    height = int(rollout["camera_height"])
    width = int(rollout["camera_width"])
    if renderer is not None:
        renderers = [renderer]
    else:
        count = max(1, min(int(n_renderers), max(1, len(sensor_seeds))))
        kind = ThreadedRenderer if count > 1 else SharedRenderer
        renderers = [kind(model, height=height, width=width) for _ in range(count)]
    envs = [
        RandomSceneEnv(
            scene_dir,
            config,
            sensor_seed,
            model=model,
            renderer=renderers[index % len(renderers)],
        )
        for index, sensor_seed in enumerate(sensor_seeds)
    ]
    return model, renderers, envs


def _unique_renderers(envs: list["RandomSceneEnv"]) -> list[SharedRenderer]:
    renderers = []
    seen: set[int] = set()
    for env in envs:
        key = id(env.renderer)
        if key in seen:
            continue
        seen.add(key)
        renderers.append(env.renderer)
    return renderers


def retarget_scene_envs(
    envs: list["RandomSceneEnv"],
    scene_dir: Path,
    sensor_seeds: list[int] | None = None,
) -> None:
    scene_dir = Path(scene_dir).resolve()
    old_models = []
    seen = set()
    old_datas = []
    old_trackers = []
    for env in envs:
        if id(env.model) not in seen:
            seen.add(id(env.model))
            old_models.append(env.model)
        old_datas.append(env.data)
        old_trackers.append(env.reward_tracker)
    model = load_scene_model(scene_dir)
    for renderer in _unique_renderers(envs):
        renderer.rebind(model)
    seeds = list(sensor_seeds) if sensor_seeds is not None else [0] * len(envs)
    for env, sensor_seed in zip(envs, seeds, strict=True):
        env.reload(scene_dir, model=model, renderer=env.renderer, sensor_seed=sensor_seed)
    del old_datas, old_trackers, old_models
    gc.collect()


class RandomSceneEnv:
    def __init__(
        self,
        scene_dir: Path,
        config: dict,
        sensor_seed: int,
        model: mujoco.MjModel | None = None,
        renderer: SharedRenderer | None = None,
    ):
        self.scene_dir = scene_dir.resolve()
        self.config = config
        self.rollout_config = config["environment"]["rollout"]
        self.scene_metadata = json.loads(
            (self.scene_dir / "metadata.json").read_text()
        )
        self.model = model or load_scene_model(self.scene_dir)
        self.data = mujoco.MjData(self.model)
        override_hz = self.rollout_config.get("control_hz")
        self.control_hz = float(
            override_hz
            if override_hz not in (None, "")
            else self.scene_metadata["control_hz"]
        )
        self.control_timestep = 1.0 / self.control_hz
        self.physics_steps = max(
            1,
            round(self.control_timestep / self.model.opt.timestep),
        )
        self.camera_names = list(self.scene_metadata["cameras"])
        self.camera_profiles = self.scene_metadata["camera_profiles"]
        self.sensor_rng = np.random.default_rng(sensor_seed)
        self.sensor_time_seconds = 0.0
        self.camera_cache: dict[str, Image.Image] = {}
        self.camera_last_capture = {
            name: float("-inf") for name in self.camera_names
        }
        self._owns_renderer = renderer is None
        self.renderer = renderer or SharedRenderer(
            self.model,
            height=int(self.rollout_config["camera_height"]),
            width=int(self.rollout_config["camera_width"]),
        )
        self.reward_tracker = None

    def reload(
        self,
        scene_dir: Path,
        model: mujoco.MjModel | None = None,
        renderer: SharedRenderer | None = None,
        sensor_seed: int | None = None,
    ) -> None:
        old_data = self.data
        old_tracker = self.reward_tracker
        self.scene_dir = Path(scene_dir).resolve()
        self.scene_metadata = json.loads(
            (self.scene_dir / "metadata.json").read_text()
        )
        self.model = model or load_scene_model(self.scene_dir)
        self.data = mujoco.MjData(self.model)
        self.reward_tracker = None
        del old_data, old_tracker
        if sensor_seed is not None:
            self.sensor_rng = np.random.default_rng(sensor_seed)
        override_hz = self.rollout_config.get("control_hz")
        self.control_hz = float(
            override_hz
            if override_hz not in (None, "")
            else self.scene_metadata["control_hz"]
        )
        self.control_timestep = 1.0 / self.control_hz
        self.physics_steps = max(
            1,
            round(self.control_timestep / self.model.opt.timestep),
        )
        self.camera_names = list(self.scene_metadata["cameras"])
        self.camera_profiles = self.scene_metadata["camera_profiles"]
        self.camera_cache.clear()
        self.camera_last_capture = {
            name: float("-inf") for name in self.camera_names
        }
        if renderer is not None:
            self.renderer = renderer
            self._owns_renderer = False

    def _reset_physics(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        home_key = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
        if home_key >= 0:
            self.data.qpos[:ROBOT_ACTION_DIM] = self.model.key_qpos[
                home_key,
                :ROBOT_ACTION_DIM,
            ]
            self.data.ctrl[:ROBOT_ACTION_DIM] = self.model.key_ctrl[
                home_key,
                :ROBOT_ACTION_DIM,
            ]
        mujoco.mj_forward(self.model, self.data)
        settle_steps = round(
            float(self.config["scene"]["settle_seconds"])
            / self.model.opt.timestep
        )
        if settle_steps:
            mujoco.mj_step(self.model, self.data, nstep=settle_steps)

    def reset(self) -> dict:
        self._reset_physics()
        self.sensor_time_seconds = 0.0
        self.camera_cache.clear()
        self.camera_last_capture = {
            name: float("-inf") for name in self.camera_names
        }
        self.reward_tracker = PrismRewardTracker(
            self.model,
            self.data,
            self.scene_metadata,
            self.config["reward"],
            float(self.rollout_config["duration_seconds"]),
        )
        return self.observation(render_images=True)

    def _camera_due(self, camera_name: str) -> bool:
        frame_period = 1.0 / float(
            self.camera_profiles[camera_name]["frame_rate_hz"]
        )
        return (
            camera_name not in self.camera_cache
            or self.sensor_time_seconds
            - self.camera_last_capture[camera_name]
            >= frame_period
        )

    def _process_capture(self, camera_name: str, raw: np.ndarray) -> Image.Image:
        image, _ = apply_camera_pipeline(
            raw,
            self.config,
            self.sensor_rng,
            self.camera_profiles[camera_name],
        )
        self.camera_cache[camera_name] = image
        self.camera_last_capture[camera_name] = self.sensor_time_seconds
        return image

    def render_camera(self, camera_name: str) -> Image.Image:
        if not self._camera_due(camera_name):
            return self.camera_cache[camera_name]
        with self.renderer.lock:
            if not self._camera_due(camera_name):
                return self.camera_cache[camera_name]
            raw = self.renderer.render_rgb(self.data, camera_name, copy=True)
        return self._process_capture(camera_name, raw)

    def render_all(self) -> dict[str, Image.Image]:
        due = [name for name in self.camera_names if self._camera_due(name)]
        raws: dict[str, np.ndarray] = {}
        if due:
            with self.renderer.lock:
                for name in due:
                    if self._camera_due(name):
                        raws[name] = self.renderer.render_rgb(
                            self.data, name, copy=True
                        )
        for name, raw in raws.items():
            self._process_capture(name, raw)
        return {
            name: self.camera_cache[name] for name in self.camera_names
        }

    def capture_due_camera_frames(
        self,
    ) -> tuple[float, dict[str, Image.Image]]:
        capture_time = self.sensor_time_seconds
        frames = {}
        due = [
            name
            for name in self.camera_names
            if self._camera_due(name)
        ]
        if due:
            with self.renderer.lock:
                for name in due:
                    frames[name] = Image.fromarray(
                        self.renderer.render_rgb(self.data, name, copy=True)
                    )
        return capture_time, frames

    def process_camera_frames(
        self,
        capture_time: float,
        frames: dict[str, Image.Image],
    ) -> list[Image.Image]:
        for camera_name in self.camera_names:
            raw_image = frames.get(camera_name)
            if raw_image is None:
                continue
            image, _ = apply_camera_pipeline(
                raw_image,
                self.config,
                self.sensor_rng,
                self.camera_profiles[camera_name],
            )
            self.camera_cache[camera_name] = image
            self.camera_last_capture[camera_name] = capture_time
        return [
            self.camera_cache[camera_name]
            for camera_name in self.camera_names
        ]

    def render_camera_raw(self, camera_name: str) -> Image.Image:
        with self.renderer.lock:
            return Image.fromarray(
                self.renderer.render_rgb(self.data, camera_name, copy=True)
            )

    def _render_images(self) -> list[Image.Image]:
        frames = self.render_all()
        return [frames[name] for name in self.camera_names]

    def observation(self, render_images: bool) -> dict:
        state = self.data.qpos[:ROBOT_ACTION_DIM].astype(np.float32).copy()
        if self.rollout_config["actions_are_degrees"]:
            state = np.rad2deg(state).astype(np.float32)
        return {
            "images": self._render_images() if render_images else [],
            "state": state.tolist(),
        }

    def step(
        self,
        action: np.ndarray,
        render_images: bool,
    ) -> tuple[dict, bool]:
        done = self.step_physics(action)
        return self.observation(render_images), done

    def step_actions(
        self,
        actions: np.ndarray,
        max_steps: int,
        start_steps: int,
        on_step=None,
    ) -> tuple[int, bool]:
        steps = int(start_steps)
        done = False
        for action in actions:
            if steps >= max_steps:
                break
            done = self.step_physics(np.asarray(action, dtype=np.float32))
            steps += 1
            if on_step is not None:
                on_step(self.data)
            if done:
                break
        return steps, done

    def step_physics(self, action: np.ndarray) -> bool:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ROBOT_ACTION_DIM,):
            raise ValueError(
                f"expected action shape {(ROBOT_ACTION_DIM,)}, got {action.shape}"
            )
        if self.rollout_config["actions_are_degrees"]:
            action = np.deg2rad(action)
        if not np.isfinite(action).all():
            action = np.zeros_like(action)
        control_range = self.model.actuator_ctrlrange[:ROBOT_ACTION_DIM]
        action = np.clip(action, control_range[:, 0], control_range[:, 1])
        self.data.ctrl[:ROBOT_ACTION_DIM] = action
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        self.reward_tracker.update(self.control_timestep)
        self.sensor_time_seconds += self.control_timestep
        return bool(self.reward_tracker.success)

    def finalize_reward(self) -> RawRolloutReward:
        return self.reward_tracker.finalize()

    def close(self) -> None:
        if self._owns_renderer:
            self.renderer.close()
        self.reward_tracker = None
        self.camera_cache.clear()
        self.data = None
        self.model = None
