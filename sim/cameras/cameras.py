import io
import math
import queue
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from core.common import uniform_range, vector

# ======Settings=========
COLLISION_GROUP = 3
COLLAGE_HERO_ORDER = ["overview", "front", "left", "right", "top", "wrist"]
COLLAGE_BORDER_PX = 5
COLLAGE_BORDER_COLOR = (255, 255, 255)
# ======Settings=========


class SharedRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        height: int,
        width: int,
    ) -> None:
        self.model = model
        self.height = int(height)
        self.width = int(width)
        self.lock = threading.Lock()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[COLLISION_GROUP] = 0
        self.renderer = mujoco.Renderer(
            model,
            height=self.height,
            width=self.width,
        )
        self.buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera_ids: dict[str, int] = {}

    def camera_id(self, camera_name: str) -> int:
        camera_id = self._camera_ids.get(camera_name)
        if camera_id is None:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
            if camera_id < 0:
                raise RuntimeError(f"unknown camera {camera_name}")
            self._camera_ids[camera_name] = camera_id
        return camera_id

    def render_rgb(
        self,
        data: mujoco.MjData,
        camera_name: str,
        copy: bool = False,
    ) -> np.ndarray:
        self._camera.fixedcamid = self.camera_id(camera_name)
        self.renderer.update_scene(
            data,
            camera=self._camera,
            scene_option=self.option,
        )
        pixels = self.renderer.render(out=self.buffer)
        if copy:
            return pixels.copy()
        return pixels

    def rebind(self, model: mujoco.MjModel) -> None:
        with self.lock:
            old_renderer = self.renderer
            old_model = self.model
            if old_renderer is not None:
                old_renderer.close()
            self.model = model
            self._camera_ids = {}
            self.renderer = mujoco.Renderer(
                model,
                height=self.height,
                width=self.width,
            )
        del old_renderer, old_model

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ThreadedRenderer:
    """EGL context lives on one thread; other threads submit render jobs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        height: int,
        width: int,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._exc: BaseException | None = None
        self._ready = threading.Event()
        self._inner: SharedRenderer | None = None
        self._thread = threading.Thread(
            target=self._loop,
            args=(model, self.height, self.width),
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._exc is not None:
            raise self._exc

    def _loop(self, model: mujoco.MjModel, height: int, width: int) -> None:
        try:
            self._inner = SharedRenderer(model, height, width)
        except BaseException as exc:
            self._exc = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                if self._inner is not None:
                    self._inner.close()
                return
            fn, args, reply = job
            try:
                reply.put(fn(*args))
            except BaseException as exc:
                reply.put(exc)

    def _call(self, fn, *args):
        reply: queue.Queue = queue.Queue()
        self._jobs.put((fn, args, reply))
        payload = reply.get()
        if isinstance(payload, BaseException):
            raise payload
        return payload

    def render_rgb(
        self,
        data: mujoco.MjData,
        camera_name: str,
        copy: bool = False,
    ) -> np.ndarray:
        inner = self._inner
        if inner is None:
            raise RuntimeError("renderer is closed")
        return self._call(inner.render_rgb, data, camera_name, copy)

    def rebind(self, model: mujoco.MjModel) -> None:
        inner = self._inner
        if inner is None:
            raise RuntimeError("renderer is closed")
        self._call(inner.rebind, model)

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5.0)


def rotation_matrix_xyz(angles: np.ndarray) -> np.ndarray:
    x, y, z = angles
    rx = np.array(
        [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    )
    ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    )
    rz = np.array(
        [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


def perturb_xyaxes(
    xyaxes: list[float], rng: np.random.Generator, rotation_jitter_deg: float
) -> list[float]:
    x_axis = np.asarray(xyaxes[:3], dtype=np.float64)
    y_axis = np.asarray(xyaxes[3:], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis -= x_axis * np.dot(x_axis, y_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    base = np.column_stack([x_axis, y_axis, z_axis])
    angles = np.deg2rad(
        rng.uniform(-rotation_jitter_deg, rotation_jitter_deg, size=3)
    )
    rotated = base @ rotation_matrix_xyz(angles)
    return [*rotated[:, 0], *rotated[:, 1]]


def look_at_xyaxes(
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float,
) -> list[float]:
    forward = target - position
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-4:
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    roll = math.radians(roll_deg)
    rotated_x = math.cos(roll) * x_axis + math.sin(roll) * y_axis
    rotated_y = -math.sin(roll) * x_axis + math.cos(roll) * y_axis
    return [*rotated_x, *rotated_y]


def clamp_camera_fovy(
    parameters: dict,
    camera_config: dict,
    camera_name: str,
) -> None:
    parameters["optical_fovy_deg"] = parameters["fovy_deg"]
    limits = camera_config["fovy_limits_by_camera"][camera_name]
    parameters["fovy_deg"] = float(
        np.clip(
            parameters["fovy_deg"],
            float(limits[0]),
            float(limits[1]),
        )
    )


def add_cameras(
    worldbody: ET.Element, config: dict, rng: np.random.Generator
) -> tuple[list[str], dict[str, dict]]:
    camera_config = config["cameras"]
    position_jitter = float(camera_config["position_jitter_m"])
    wrist_position_jitter = float(camera_config["wrist_position_jitter_m"])
    wrist_rotation_jitter = float(camera_config["wrist_rotation_jitter_deg"])
    roll_jitter = float(camera_config["roll_jitter_deg"])
    required = list(camera_config["required"])
    total_count = int(
        rng.integers(
            int(camera_config["count"][0]),
            int(camera_config["count"][1]) + 1,
        )
    )
    optional = [name for name in camera_config["mounts"] if name not in required]
    optional_count = total_count - len(required)
    selected_optional = (
        [
            str(name)
            for name in rng.choice(
                optional,
                size=optional_count,
                replace=False,
            )
        ]
        if optional_count
        else []
    )
    camera_names = required + selected_optional
    camera_profiles = {}
    frame_rate = uniform_range(rng, camera_config["frame_rate_hz"])

    wrist = worldbody.find(".//camera[@name='wrist']")
    if wrist is None:
        raise RuntimeError("robot model has no wrist camera")
    wrist_position = np.asarray([float(value) for value in wrist.get("pos").split()])
    wrist_axes = [float(value) for value in wrist.get("xyaxes").split()]
    wrist.set(
        "pos",
        vector(
            wrist_position
            + rng.uniform(
                -wrist_position_jitter,
                wrist_position_jitter,
                size=3,
            )
        ),
    )
    wrist.set(
        "xyaxes",
        vector(perturb_xyaxes(wrist_axes, rng, wrist_rotation_jitter)),
    )
    profile_name, parameters = sample_camera_profile(
        config,
        rng,
        subject_distance_m=0.25,
        frame_rate=frame_rate,
    )
    clamp_camera_fovy(parameters, camera_config, "wrist")
    camera_profiles["wrist"] = {"profile": profile_name, **parameters}
    wrist.set(
        "fovy",
        f"{parameters['fovy_deg']:.9g}",
    )

    for name in camera_names:
        if name == "wrist":
            continue
        mount = camera_config["mounts"][name]
        position = np.asarray(mount["position_m"], dtype=np.float64)
        position += rng.uniform(-position_jitter, position_jitter, size=3)
        target = np.asarray(camera_config["look_at_m"], dtype=np.float64)
        target += rng.uniform(
            -np.asarray(camera_config["look_at_jitter_m"], dtype=np.float64),
            np.asarray(camera_config["look_at_jitter_m"], dtype=np.float64),
        )
        xyaxes = look_at_xyaxes(
            position,
            target,
            rng.uniform(-roll_jitter, roll_jitter),
        )
        profile_name, parameters = sample_camera_profile(
            config,
            rng,
            subject_distance_m=float(np.linalg.norm(position - target)),
            frame_rate=frame_rate,
        )
        clamp_camera_fovy(parameters, camera_config, name)
        camera_profiles[name] = {"profile": profile_name, **parameters}
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": name,
                "pos": vector(position),
                "xyaxes": vector(xyaxes),
                "fovy": f"{parameters['fovy_deg']:.9g}",
            },
        )
    return camera_names, camera_profiles


def sample_camera_profile(
    config: dict,
    rng: np.random.Generator,
    subject_distance_m: float = 0.5,
    frame_rate: float | None = None,
) -> tuple[str, dict]:
    camera_config = config["cameras"]
    profiles = camera_config["profiles"]
    names = list(profiles)
    probabilities = np.asarray(
        [float(profiles[name]["probability"]) for name in names]
    )
    probabilities /= probabilities.sum()
    profile_name = str(rng.choice(names, p=probabilities))
    profile = profiles[profile_name]
    integer_parameters = {"bit_depth", "jpeg_quality"}
    sampled = {}
    for name, values in profile.items():
        if name == "probability":
            continue
        if name in integer_parameters:
            sampled[name] = int(
                rng.integers(int(values[0]), int(values[1]) + 1)
            )
        else:
            sampled[name] = uniform_range(rng, values)

    if frame_rate is None:
        frame_rate = uniform_range(rng, camera_config["frame_rate_hz"])
    exposure_time = (
        sampled["shutter_fraction_of_frame"] / frame_rate
    )
    iso_gain = sampled["iso"] / 100.0
    shot_noise_photons = max(
        sampled["full_well_electrons"] / max(iso_gain, 1.0),
        80.0,
    )
    read_noise_std = min(
        sampled["read_noise_electrons"]
        / sampled["full_well_electrons"]
        * iso_gain,
        0.015,
    )
    motion_blur_px = min(
        sampled["camera_motion_px_s"] * exposure_time,
        2.5,
    )
    rolling_shutter_shift_px = min(
        sampled["camera_motion_px_s"]
        * sampled["rolling_shutter_readout_ms"]
        / 1000.0,
        2.0,
    )
    focus_error = (
        abs(sampled["focus_distance_m"] - subject_distance_m)
        / max(sampled["focus_distance_m"], subject_distance_m, 0.1)
    )
    defocus_blur_px = min(
        sampled["defocus_scale_px"]
        * focus_error
        / sampled["f_number"],
        0.9,
    )
    fovy = math.degrees(
        2.0
        * math.atan(
            sampled["sensor_height_mm"]
            / (2.0 * sampled["focal_length_mm"])
        )
    )
    fovy = float(
        np.clip(
            fovy,
            float(camera_config["fovy_clamp_deg"][0]),
            float(camera_config["fovy_clamp_deg"][1]),
        )
    )
    temperature_offset = (
        sampled["white_balance_temperature_k"] - 5500.0
    ) / 3000.0
    white_balance_red = float(
        np.clip(
            1.0 + temperature_offset * sampled["auto_white_balance_error"],
            0.88,
            1.12,
        )
    )
    white_balance_blue = float(
        np.clip(
            1.0 - temperature_offset * sampled["auto_white_balance_error"],
            0.88,
            1.12,
        )
    )
    parameters = {
        **sampled,
        "frame_rate_hz": frame_rate,
        "subject_distance_m": subject_distance_m,
        "fovy_deg": fovy,
        "exposure_time_s": exposure_time,
        "shot_noise_photons": shot_noise_photons,
        "read_noise_std": read_noise_std,
        "motion_blur_px": motion_blur_px,
        "motion_blur_angle_deg": float(rng.uniform(0.0, 180.0)),
        "rolling_shutter_shift_px": rolling_shutter_shift_px,
        "defocus_blur_px": defocus_blur_px,
        "effective_resolution_scale": sampled["native_resolution_scale"],
        "exposure_ev": sampled["exposure_compensation_ev"],
        "white_balance_red": white_balance_red,
        "white_balance_blue": white_balance_blue,
    }
    return profile_name, parameters


def bilinear_sample(
    image: np.ndarray, source_x: np.ndarray, source_y: np.ndarray
) -> np.ndarray:
    height, width = image.shape[:2]
    source_x = np.clip(source_x, 0.0, width - 1.0)
    source_y = np.clip(source_y, 0.0, height - 1.0)
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (source_x - x0)[..., None]
    wy = (source_y - y0)[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def apply_optical_distortion(image: np.ndarray, parameters: dict) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    scale = max(cx, cy, 1.0)
    x = (xx - cx) / scale
    y = (yy - cy) / scale
    radius_squared = x * x + y * y
    k1 = float(parameters["radial_distortion_k1"])
    tangential = float(parameters["tangential_distortion"])
    radial = 1.0 + k1 * radius_squared
    source_x = (
        x * radial
        + 2.0 * tangential * x * y
        + tangential * (radius_squared + 2.0 * x * x)
    )
    source_y = (
        y * radial
        + tangential * (radius_squared + 2.0 * y * y)
        + 2.0 * tangential * x * y
    )
    rolling_shift = float(parameters["rolling_shutter_shift_px"])
    distorted = bilinear_sample(
        image,
        source_x * scale + cx + rolling_shift * y,
        source_y * scale + cy,
    )

    aberration = float(parameters["chromatic_aberration_px"])
    if aberration > 0.0:
        normalized_shift = aberration / scale
        red = bilinear_sample(
            image,
            x * (1.0 + normalized_shift) * scale + cx,
            y * (1.0 + normalized_shift) * scale + cy,
        )[..., 0]
        blue = bilinear_sample(
            image,
            x * (1.0 - normalized_shift) * scale + cx,
            y * (1.0 - normalized_shift) * scale + cy,
        )[..., 2]
        distorted[..., 0] = red
        distorted[..., 2] = blue
    return distorted


def apply_motion_blur(
    image: np.ndarray, length: int, angle_deg: float
) -> np.ndarray:
    if length <= 1:
        return image
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    angle = math.radians(angle_deg)
    offsets = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    blurred = np.zeros_like(image, dtype=np.float64)
    for offset in offsets:
        blurred += bilinear_sample(
            image,
            xx + math.cos(angle) * offset,
            yy + math.sin(angle) * offset,
        )
    return blurred / len(offsets)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.maximum(image, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def apply_fast_camera_pipeline(
    image: Image.Image | np.ndarray,
    parameters: dict,
    rng: np.random.Generator,
) -> Image.Image:
    if isinstance(image, Image.Image):
        pixels = np.asarray(image.convert("RGB"))
    else:
        pixels = image
    blur_radius = float(parameters["defocus_blur_px"])
    if blur_radius > 0.05:
        pixels = np.asarray(
            Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(blur_radius))
        )
    array = np.multiply(pixels, np.float32(1.0 / 255.0), dtype=np.float32)
    array *= np.float32(2.0 ** float(parameters["exposure_ev"]))
    array[..., 0] *= np.float32(parameters["white_balance_red"])
    array[..., 2] *= np.float32(parameters["white_balance_blue"])
    np.clip(array, 0.0, 1.0, out=array)
    photons = max(float(parameters["shot_noise_photons"]), 1.0)
    read_noise = float(parameters["read_noise_std"])
    noise = rng.standard_normal(size=array.shape, dtype=np.float32)
    noise *= np.sqrt(array / photons + read_noise * read_noise)
    array += noise
    np.clip(array, 0.0, 1.0, out=array)
    array **= np.float32(1.0 / float(parameters["gamma"]))
    np.clip(array, 0.0, 1.0, out=array)
    contrast = float(parameters["contrast"])
    if abs(contrast - 1.0) > 1e-3:
        array -= 0.5
        array *= np.float32(contrast)
        array += 0.5
        np.clip(array, 0.0, 1.0, out=array)
    saturation = float(parameters["saturation"])
    if abs(saturation - 1.0) > 1e-3:
        gray = array.mean(axis=2, keepdims=True)
        array -= gray
        array *= np.float32(saturation)
        array += gray
        np.clip(array, 0.0, 1.0, out=array)
    levels = 2 ** int(parameters["bit_depth"]) - 1
    np.rint(array * levels, out=array)
    array /= levels
    sharpness = float(parameters["sharpness"])
    image = Image.fromarray(np.uint8(array * 255.0))
    if abs(sharpness - 1.0) > 1e-3:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    resolution_scale = float(parameters["effective_resolution_scale"])
    if resolution_scale < 0.999:
        width, height = image.size
        reduced_size = (
            max(16, round(width * resolution_scale)),
            max(16, round(height * resolution_scale)),
        )
        image = image.resize(reduced_size, Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def apply_camera_pipeline(
    image: Image.Image | np.ndarray,
    config: dict,
    rng: np.random.Generator,
    profile_parameters: dict | None = None,
) -> tuple[Image.Image, dict]:
    if profile_parameters is None:
        profile_name, parameters = sample_camera_profile(config, rng)
    else:
        profile_name = str(profile_parameters["profile"])
        parameters = {
            name: value
            for name, value in profile_parameters.items()
            if name != "profile"
        }
    if config["cameras"]["fast_pipeline"]:
        return (
            apply_fast_camera_pipeline(image, parameters, rng),
            {"profile": profile_name, **parameters},
        )
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    else:
        array = np.asarray(image, dtype=np.float64) / 255.0
    array = apply_optical_distortion(array, parameters)
    array = apply_motion_blur(
        array,
        int(round(parameters["motion_blur_px"])),
        float(parameters["motion_blur_angle_deg"]),
    )
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0))
    image = image.filter(
        ImageFilter.GaussianBlur(float(parameters["defocus_blur_px"]))
    )

    linear = srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)
    linear *= 2.0 ** float(parameters["exposure_ev"])
    linear[..., 0] *= float(parameters["white_balance_red"])
    linear[..., 2] *= float(parameters["white_balance_blue"])
    height, width = linear.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    radius = np.sqrt(
        ((xx - (width - 1) / 2.0) / max((width - 1) / 2.0, 1.0)) ** 2
        + ((yy - (height - 1) / 2.0) / max((height - 1) / 2.0, 1.0)) ** 2
    )
    vignette = np.clip(
        1.0 - float(parameters["vignette"]) * radius**2,
        0.15,
        1.0,
    )
    linear *= vignette[..., None]
    photons = float(parameters["shot_noise_photons"])
    linear = rng.poisson(np.clip(linear, 0.0, 1.0) * photons) / photons
    linear += rng.normal(
        0.0,
        float(parameters["read_noise_std"]),
        size=linear.shape,
    )
    noise_floor = 2.0 ** (-float(parameters["dynamic_range_stops"]))
    linear = (linear - noise_floor) / (1.0 - noise_floor)
    array = np.clip(
        linear_to_srgb(np.clip(linear, 0.0, 1.0)),
        0.0,
        1.0,
    )
    array = np.clip(
        array ** (1.0 / float(parameters["gamma"])),
        0.0,
        1.0,
    )

    bit_depth = int(parameters["bit_depth"])
    levels = 2**bit_depth - 1
    array = np.rint(array * levels) / levels
    dead_probability = float(parameters["dead_pixel_probability"])
    dead_pixels = rng.random((height, width)) < dead_probability
    if np.any(dead_pixels):
        array[dead_pixels] = rng.choice(
            [0.0, 1.0],
            size=(int(dead_pixels.sum()), 1),
        )
    image = Image.fromarray(np.uint8(array * 255.0))
    image = ImageEnhance.Contrast(image).enhance(float(parameters["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(parameters["saturation"]))
    image = ImageEnhance.Sharpness(image).enhance(
        float(parameters["sharpness"])
    )

    resolution_scale = float(parameters["effective_resolution_scale"])
    if resolution_scale < 0.999:
        reduced_size = (
            max(16, round(width * resolution_scale)),
            max(16, round(height * resolution_scale)),
        )
        image = image.resize(reduced_size, Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(parameters["jpeg_quality"]),
        subsampling=2,
    )
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return image, {"profile": profile_name, **parameters}


def select_collage_cameras(images: dict[str, Image.Image]) -> list[str]:
    ordered = [name for name in COLLAGE_HERO_ORDER if name in images]
    ordered.extend(name for name in images if name not in ordered)
    return ordered[:3]


def save_camera_collage(
    images: dict[str, Image.Image],
    camera_metadata: dict[str, dict],
    output_path: Path,
) -> None:
    names = select_collage_cameras(images)
    hero = images[names[0]]
    width, height = hero.size
    border = COLLAGE_BORDER_PX
    if len(names) >= 3:
        collage = Image.new(
            "RGB",
            (width * 3 + border * 3, height * 2 + border * 3),
            COLLAGE_BORDER_COLOR,
        )
        collage.paste(
            hero.resize(
                (width * 2, height * 2 + border),
                Image.Resampling.LANCZOS,
            ),
            (border, border),
        )
        collage.paste(
            images[names[1]].resize((width, height), Image.Resampling.LANCZOS),
            (width * 2 + border * 2, border),
        )
        collage.paste(
            images[names[2]].resize((width, height), Image.Resampling.LANCZOS),
            (width * 2 + border * 2, height + border * 2),
        )
    elif len(names) == 2:
        collage = Image.new(
            "RGB",
            (width * 2 + border * 3, height + border * 2),
            COLLAGE_BORDER_COLOR,
        )
        collage.paste(hero, (border, border))
        collage.paste(
            images[names[1]].resize((width, height), Image.Resampling.LANCZOS),
            (width + border * 2, border),
        )
    else:
        collage = hero
    collage.save(output_path)


def render_cameras(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_dir: Path,
    config: dict,
    camera_names: list[str],
    camera_profiles: dict[str, dict],
    rng: np.random.Generator,
) -> dict[str, dict]:
    width = int(config["environment"]["preview"]["render_width"])
    height = int(config["environment"]["preview"]["render_height"])
    renderer = SharedRenderer(model, height=height, width=width)
    images = {}
    camera_metadata = {}
    try:
        for camera_name in camera_names:
            raw = renderer.render_rgb(data, camera_name)
            image, metadata = apply_camera_pipeline(
                raw,
                config,
                rng,
                camera_profiles[camera_name],
            )
            image.save(output_dir / f"{camera_name}.png")
            images[camera_name] = image
            camera_metadata[camera_name] = metadata
    finally:
        renderer.close()
    save_camera_collage(
        images,
        camera_metadata,
        output_dir / "cameras.png",
    )
    return camera_metadata
