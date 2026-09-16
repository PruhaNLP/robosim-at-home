import gc
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

# ======Settings=========
DEFAULT_EGL_DEVICE_ID = "0"
MAX_LAYOUT_ATTEMPTS = 1
OFFSCREEN_SIZE_PX = 2048
# ======Settings=========

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "MUJOCO_EGL_DEVICE_ID",
    os.environ.get("ROBOSIM_RENDER_EGL_DEVICE_ID", DEFAULT_EGL_DEVICE_ID),
)

import mujoco
import numpy as np
from PIL import Image

from cameras.cameras import add_cameras, render_cameras
from core.common import jitter, resolve_path, uniform_range, vector, xml_path
from core.config import (
    GENERATE_SCENE_DIR,
    SIM_ROOT,
    load_config,
    normalize_policy_mode,
    resolve_stored,
)
from language.prompts import generate_prompt
from scene.lighting import add_lights
from scene.objects import (
    add_object_assets_and_body,
    make_scene_object,
    mesh_bounds,
    object_index,
    sample_position,
    selected_object_dirs,
)
from scene.physics import randomize_robot_physics
from scene.rooms import add_room_skybox, choose_room
from scene.world import add_table_and_tray


def disable_reflections(root: ET.Element) -> None:
    for material in root.findall(".//material"):
        material.set("reflectance", "0")
        material.set("specular", "0")
        material.set("shininess", "0")
    for light in root.findall(".//light"):
        light.set("specular", "0 0 0")
    for headlight in root.findall(".//headlight"):
        headlight.set("specular", "0 0 0")


def generate(
    seed_override: int | None = None,
    output_dir_override: Path | None = None,
    write_previews: bool = True,
) -> dict:
    config = load_config()
    seed = int(config["seed"] if seed_override is None else seed_override)
    rng = np.random.default_rng(seed)
    config_dir = SIM_ROOT
    targets_dir = resolve_path(config_dir, config["paths"]["targets_dir"])
    distractors_dir = resolve_path(
        config_dir,
        config["paths"]["distractors_dir"],
    )
    table_textures_dir = resolve_path(
        config_dir,
        config["paths"]["table_textures_dir"],
    )
    rooms_dir = resolve_path(config_dir, config["paths"]["rooms_dir"])
    robot_xml = resolve_path(config_dir, config["paths"]["robot_xml"])
    output_dir = (
        resolve_stored(output_dir_override)
        if output_dir_override is not None
        else GENERATE_SCENE_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = object_index(targets_dir, distractors_dir)
    target_pool = selected_object_dirs(
        catalog,
        config["scene"].get("targets"),
    )
    distractor_pool = selected_object_dirs(
        catalog,
        config["scene"].get("distractors"),
    )
    table_textures = sorted(
        path
        for path in table_textures_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not target_pool or not distractor_pool or not table_textures:
        raise RuntimeError("targets, distractors, and table textures must not be empty")
    if normalize_policy_mode(config.get("policy_mode")) == "act" and len(target_pool) != 1:
        raise RuntimeError("ACT needs exactly one target object")
    distractor_max = min(
        int(config["scene"]["distractors"]["max"]),
        len(distractor_pool),
    )
    distractor_min = min(
        int(config["scene"]["distractors"]["min"]),
        distractor_max,
    )
    target_dir = target_pool[int(rng.integers(len(target_pool)))]
    distractor_count = int(
        rng.integers(
            distractor_min,
            distractor_max + 1,
        )
    )
    distractor_indices = rng.choice(
        len(distractor_pool),
        size=distractor_count,
        replace=False,
    )
    selected_distractors = [
        distractor_pool[int(index)] for index in distractor_indices
    ]
    rooms_enabled = bool(config["scene"]["rooms"].get("enabled", True))
    room_name = config["scene"]["rooms"].get("name")
    if rooms_enabled and room_name:
        room_dir = rooms_dir / str(room_name)
        if not (room_dir / "skybox.png").is_file():
            raise RuntimeError(f"unknown room {room_name}")
    elif rooms_enabled:
        room_dir = choose_room(rooms_dir, rng)
    else:
        room_dir = None
    table = config["scene"]["table"]
    texture_name = table.get("texture")
    if texture_name:
        table_texture_source = table_textures_dir / str(texture_name)
        if not table_texture_source.is_file():
            raise RuntimeError(f"unknown table texture {texture_name}")
    else:
        table_texture_source = table_textures[
            int(rng.integers(len(table_textures)))
        ]
    table_texture = output_dir / "table_texture.png"
    with Image.open(table_texture_source) as source:
        source.convert("RGB").save(table_texture)

    table_size = np.array(
        [
            uniform_range(rng, table["width_m"]),
            uniform_range(rng, table["depth_m"]),
        ],
        dtype=np.float64,
    )
    spawn_area_size = np.asarray(
        table["spawn_area_size_m"],
        dtype=np.float64,
    )
    spawn_area_center = np.asarray(
        table["spawn_area_center_xy_m"],
        dtype=np.float64,
    )
    margin = float(table["edge_margin_m"])
    table_center = np.array(
        [
            spawn_area_center[0],
            float(table["near_edge_y_m"]) - table_size[1] / 2.0,
        ]
    )
    table_bounds = (
        spawn_area_center[0] - spawn_area_size[0] / 2.0 + margin,
        spawn_area_center[0] + spawn_area_size[0] / 2.0 - margin,
        spawn_area_center[1] - spawn_area_size[1] / 2.0 + margin,
        spawn_area_center[1] + spawn_area_size[1] / 2.0 - margin,
    )
    robot_exclusion = (
        -float(config["scene"]["robot_exclusion_half_width_m"]),
        float(config["scene"]["robot_exclusion_half_width_m"]),
        float(table["near_edge_y_m"])
        - float(config["scene"]["robot_exclusion_depth_m"]),
        float(table["near_edge_y_m"]),
    )
    tray_config = config["scene"]["tray"]
    tray_dimensions = {
        "inner_width": uniform_range(rng, tray_config["inner_width_m"]),
        "inner_depth": uniform_range(rng, tray_config["inner_depth_m"]),
        "floor_thickness": uniform_range(
            rng,
            tray_config["floor_thickness_m"],
        ),
        "wall_thickness": uniform_range(
            rng,
            tray_config["wall_thickness_m"],
        ),
        "wall_height": uniform_range(rng, tray_config["wall_height_m"]),
    }
    tray_half_width = (
        tray_dimensions["inner_width"] / 2.0
        + tray_dimensions["wall_thickness"]
    )
    tray_half_depth = (
        tray_dimensions["inner_depth"] / 2.0
        + tray_dimensions["wall_thickness"]
    )
    gap = float(config["scene"]["minimum_object_gap_m"])
    object_inputs = [("target", target_dir)] + [
        (f"distractor_{index}", path)
        for index, path in enumerate(selected_distractors)
    ]
    bounds_by_role = {
        role: mesh_bounds(object_dir)
        for role, object_dir in object_inputs
    }

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        tray_x_low = table_bounds[0] + tray_half_width
        tray_x_high = table_bounds[1] - tray_half_width
        tray_y_low = table_bounds[2] + tray_half_depth
        tray_y_high = min(
            table_bounds[3] - tray_half_depth,
            robot_exclusion[2] - tray_half_depth,
        )
        if tray_x_low >= tray_x_high or tray_y_low >= tray_y_high:
            raise RuntimeError(
                "Tray does not fit: increase spawn area or reduce tray dimensions."
            )
        tray_center_cfg = tray_config.get("center_xy_m")
        if tray_center_cfg is not None:
            tray_center = np.asarray(tray_center_cfg, dtype=np.float64)
        else:
            tray_center = np.array(
                [
                    rng.uniform(tray_x_low, tray_x_high),
                    rng.uniform(tray_y_low, tray_y_high),
                ]
            )
        tray_rectangle = (
            tray_center[0] - tray_half_width,
            tray_center[0] + tray_half_width,
            tray_center[1] - tray_half_depth,
            tray_center[1] + tray_half_depth,
        )
        proposals = []
        for role, object_dir in object_inputs:
            max_size_range = (
                config["scene"]["target_max_size_m"]
                if role == "target"
                else config["scene"]["distractor_max_size_m"]
            )
            max_size = uniform_range(rng, max_size_range)
            bounds = bounds_by_role[role]
            raw_size = np.ptp(bounds, axis=0)
            scaled_size = raw_size * (max_size / float(np.max(raw_size)))
            radius = (
                0.5
                * math.hypot(
                    float(scaled_size[0]),
                    float(scaled_size[1]),
                )
                + gap
            )
            proposals.append(
                {
                    "role": role,
                    "object_dir": object_dir,
                    "max_size": max_size,
                    "radius": radius,
                }
            )

        occupied = []
        placements = {}
        try:
            for proposal in sorted(
                proposals,
                key=lambda item: item["radius"],
                reverse=True,
            ):
                xy = sample_position(
                    rng,
                    proposal["radius"],
                    table_bounds,
                    occupied,
                    [tray_rectangle, robot_exclusion],
                )
                occupied.append((xy, proposal["radius"]))
                placements[proposal["role"]] = xy
        except RuntimeError as error:
            raise RuntimeError(
                "Cannot place "
                f"{len(object_inputs)} objects in the {spawn_area_size[0]:g}×"
                f"{spawn_area_size[1]:g} m spawn area with gap {gap:g} m. "
                "Reduce object size, gap, or distractor count."
            ) from error
        break

    scene_objects = []
    proposals_by_role = {
        proposal["role"]: proposal for proposal in proposals
    }
    for role, object_dir in object_inputs:
        proposal = proposals_by_role[role]
        scene_objects.append(
            make_scene_object(
                object_dir,
                role,
                "target" if role == "target" else role,
                proposal["max_size"],
                placements[role],
                config,
                rng,
                yaw=0.0,
            )
        )

    root = ET.parse(robot_xml).getroot()
    root.set("model", "grpo_random_scene")
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("robot XML must contain a compiler element")
    compiler.set("meshdir", xml_path(robot_xml.parent / "assets", output_dir))
    option = root.find("option")
    gravity = jitter(
        rng,
        float(config["scene"]["physics"]["gravity_m_s2"]),
        float(config["scene"]["physics"]["relative_variation"]),
    )
    option.set("gravity", vector([0.0, 0.0, -gravity]))
    option.set(
        "timestep",
        f"{float(config['scene']['timestep_seconds']):.9g}",
    )
    robot_physics_scales = randomize_robot_physics(
        root,
        rng,
        float(config["scene"]["physics"]["relative_variation"]),
    )

    asset = root.find("asset")
    worldbody = root.find("worldbody")
    off_w = [int(config["environment"]["rollout"]["camera_width"])]
    off_h = [int(config["environment"]["rollout"]["camera_height"])]
    if write_previews:
        off_w.extend(
            (
                OFFSCREEN_SIZE_PX,
                int(config["environment"]["preview"]["render_width"]),
            )
        )
        off_h.extend(
            (
                OFFSCREEN_SIZE_PX,
                int(config["environment"]["preview"]["render_height"]),
            )
        )
    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "global",
        {
            "offwidth": str(max(off_w)),
            "offheight": str(max(off_h)),
        },
    )
    light_config = config["scene"]["lights"]
    headlight_ambient = uniform_range(
        rng,
        light_config["headlight_ambient"],
    )
    headlight_diffuse = uniform_range(
        rng,
        light_config["headlight_diffuse"],
    )
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": vector([headlight_ambient] * 3),
            "diffuse": vector([headlight_diffuse] * 3),
            "specular": "0.1 0.1 0.1",
        },
    )
    root.insert(list(root).index(asset), visual)

    if rooms_enabled and room_dir is not None:
        add_room_skybox(asset, room_dir, output_dir)
    appearance = add_table_and_tray(
        asset,
        worldbody,
        config,
        table_texture,
        table_size,
        table_center,
        tray_center,
        tray_dimensions,
        rng,
        output_dir,
    )
    lights = add_lights(worldbody, config, table_center, rng)
    camera_names, camera_profiles = add_cameras(worldbody, config, rng)
    for scene_object in scene_objects:
        add_object_assets_and_body(
            asset,
            worldbody,
            scene_object,
            resolve_stored(scene_object.source_dir),
            config["scene"]["physics"],
            rng,
            output_dir,
            robot_xml.parent / "assets",
        )
    ET.indent(root, space="  ")
    scene_xml = output_dir / "scene.xml"
    ET.ElementTree(root).write(scene_xml, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetData(model, data)
        home_key = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )
        if home_key >= 0:
            data.qpos[:6] = model.key_qpos[home_key, :6]
            data.ctrl[:6] = model.key_ctrl[home_key, :6]
            mujoco.mj_forward(model, data)
        settle_steps = round(
            float(config["scene"]["settle_seconds"]) / model.opt.timestep
        )
        if settle_steps:
            mujoco.mj_step(model, data, nstep=settle_steps)

        if write_previews:
            camera_profiles = render_cameras(
                model,
                data,
                output_dir,
                config,
                camera_names,
                camera_profiles,
                rng,
            )

        settled_positions = {}
        for scene_object in scene_objects:
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                scene_object.body_name,
            )
            settled_positions[scene_object.body_name] = np.asarray(
                data.xpos[body_id]
            ).tolist()
        prompt_metadata = {
            "target": target_dir.name,
            "appearance": appearance,
        }
        metadata = {
            "seed": seed,
            "instruction": generate_prompt(prompt_metadata, config, rng),
            "target": target_dir.name,
            "distractors": [
                path.name for path in selected_distractors
            ],
            "room": room_dir.name if room_dir is not None else None,
            "room_skybox": (
                xml_path(room_dir / "skybox.png", output_dir)
                if room_dir is not None
                else None
            ),
            "table_size_m": table_size.tolist(),
            "spawn_area_size_m": spawn_area_size.tolist(),
            "spawn_area_center_xy_m": spawn_area_center.tolist(),
            "tray": {
                **tray_dimensions,
                "center_xy": tray_center.tolist(),
            },
            "appearance": appearance,
            "table_texture_source": xml_path(table_texture_source, output_dir),
            "lights": lights,
            "cameras": camera_names,
            "camera_profiles": camera_profiles,
            "control_hz": float(config["environment"]["rollout"]["control_hz"]),
            "gravity_m_s2": gravity,
            "robot_physics_multipliers": robot_physics_scales,
            "objects": [
                asdict(scene_object) for scene_object in scene_objects
            ],
            "settled_positions": settled_positions,
            "object_physics_note": (
                "GSO has no measured mass/friction; configured nominal "
                "estimates are randomized by ±5%."
            ),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )
        return metadata
    finally:
        del data
        del model
        gc.collect()
