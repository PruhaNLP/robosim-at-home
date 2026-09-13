import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.common import jitter_vector, uniform_range, vector, xml_path

# ======Settings=========
# World ranges are defined in scene/config.yaml.
# ======Settings=========


def add_table_and_tray(
    asset: ET.Element,
    worldbody: ET.Element,
    config: dict,
    table_texture: Path,
    table_size: np.ndarray,
    table_center: np.ndarray,
    tray_center: np.ndarray,
    tray_dimensions: dict,
    rng: np.random.Generator,
    origin: Path,
) -> dict:
    table = config["scene"]["table"]
    appearance = config["scene"]["appearance"]
    physics = config["scene"]["physics"]
    variation = float(physics["relative_variation"])
    table_repeat = uniform_range(rng, appearance["table_texture_repeat"])
    reflectance = uniform_range(rng, appearance["table_reflectance"])
    specular = uniform_range(rng, appearance["table_specular"])
    shininess = uniform_range(rng, appearance["table_shininess"])

    ET.SubElement(
        asset,
        "texture",
        {
            "name": "table_texture",
            "type": "2d",
            "file": xml_path(table_texture, origin),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "table_material",
            "texture": "table_texture",
            "texuniform": "true",
            "texrepeat": vector([table_repeat, table_repeat]),
            "reflectance": f"{reflectance:.9g}",
            "specular": f"{specular:.9g}",
            "shininess": f"{shininess:.9g}",
        },
    )
    tray_rgb = rng.uniform(
        np.asarray(appearance["tray_rgb"]["min"]),
        np.asarray(appearance["tray_rgb"]["max"]),
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "tray_material", "rgba": vector([*tray_rgb, 1.0])},
    )

    thickness = float(table["thickness_m"])
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "table",
            "type": "box",
            "size": vector(
                [table_size[0] / 2.0, table_size[1] / 2.0, thickness / 2.0]
            ),
            "pos": vector(
                [table_center[0], table_center[1], -thickness / 2.0]
            ),
            "material": "table_material",
            "friction": vector(
                jitter_vector(rng, physics["table_friction"], variation)
            ),
        },
    )

    inner_w = tray_dimensions["inner_width"]
    inner_d = tray_dimensions["inner_depth"]
    floor_t = tray_dimensions["floor_thickness"]
    wall_t = tray_dimensions["wall_thickness"]
    wall_h = tray_dimensions["wall_height"]
    tray = ET.SubElement(
        worldbody,
        "body",
        {"name": "tray", "pos": vector([tray_center[0], tray_center[1], 0.0])},
    )
    tray_geoms = [
        (
            "tray_floor",
            [inner_w / 2.0 + wall_t, inner_d / 2.0 + wall_t, floor_t / 2.0],
            [0, 0, floor_t / 2.0],
        ),
        (
            "tray_wall_n",
            [inner_w / 2.0 + wall_t, wall_t / 2.0, wall_h / 2.0],
            [0, inner_d / 2.0 + wall_t / 2.0, floor_t + wall_h / 2.0],
        ),
        (
            "tray_wall_s",
            [inner_w / 2.0 + wall_t, wall_t / 2.0, wall_h / 2.0],
            [0, -inner_d / 2.0 - wall_t / 2.0, floor_t + wall_h / 2.0],
        ),
        (
            "tray_wall_e",
            [wall_t / 2.0, inner_d / 2.0, wall_h / 2.0],
            [inner_w / 2.0 + wall_t / 2.0, 0, floor_t + wall_h / 2.0],
        ),
        (
            "tray_wall_w",
            [wall_t / 2.0, inner_d / 2.0, wall_h / 2.0],
            [-inner_w / 2.0 - wall_t / 2.0, 0, floor_t + wall_h / 2.0],
        ),
    ]
    for name, size, position in tray_geoms:
        ET.SubElement(
            tray,
            "geom",
            {
                "name": name,
                "type": "box",
                "size": vector(size),
                "pos": vector(position),
                "material": "tray_material",
                "friction": vector(
                    jitter_vector(rng, physics["table_friction"], variation)
                ),
            },
        )

    return {
        "texture": xml_path(table_texture, origin),
        "repeat": table_repeat,
        "reflectance": reflectance,
        "specular": specular,
        "shininess": shininess,
        "tray_rgb": tray_rgb.tolist(),
    }
