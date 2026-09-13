import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.common import SceneObject, jitter, jitter_vector, vector, xml_path
from core.config import stored_path

# ======Settings=========
MAX_PLACEMENT_ATTEMPTS = 20_000
ROBOT_VISUAL_GROUP = 2
COLLISION_GROUP = 3
# ======Settings=========


def list_object_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir() if (path / "model.xml").is_file()
    )


def object_index(targets_dir: Path, distractors_dir: Path) -> dict[str, Path]:
    index = {}
    for path in list_object_dirs(distractors_dir):
        index[path.name] = path
    for path in list_object_dirs(targets_dir):
        index[path.name] = path
    return index


def selected_object_dirs(
    pool: list[Path] | dict[str, Path],
    selection: dict | None,
) -> list[Path]:
    if isinstance(pool, dict):
        by_name = pool
        ordered = [by_name[name] for name in sorted(by_name)]
    else:
        ordered = list(pool)
        by_name = {path.name: path for path in ordered}
    selection = selection or {}
    if selection.get("all", True):
        return ordered
    include = [str(name) for name in (selection.get("include") or []) if name in by_name]
    if not include:
        raise RuntimeError("no selected objects exist in the pool")
    return [by_name[name] for name in include]


def mesh_bounds(object_dir: Path) -> np.ndarray:
    vertices = []
    with (object_dir / "model.obj").open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 4 and fields[0] == "v":
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise ValueError(f"no vertices in {object_dir / 'model.obj'}")
    array = np.asarray(vertices, dtype=np.float64)
    return np.stack([array.min(axis=0), array.max(axis=0)])


def sample_position(
    rng: np.random.Generator,
    radius: float,
    table_bounds: tuple[float, float, float, float],
    occupied: list[tuple[np.ndarray, float]],
    forbidden_rectangles: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = table_bounds
    forbidden_rectangles = forbidden_rectangles or []
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        position = np.array(
            [
                rng.uniform(x_min + radius, x_max - radius),
                rng.uniform(y_min + radius, y_max - radius),
            ]
        )
        clear_of_objects = all(
            np.linalg.norm(position - center) >= radius + other_radius
            for center, other_radius in occupied
        )
        clear_of_rectangles = all(
            math.hypot(
                position[0] - np.clip(position[0], rectangle[0], rectangle[1]),
                position[1] - np.clip(position[1], rectangle[2], rectangle[3]),
            )
            >= radius
            for rectangle in forbidden_rectangles
        )
        if clear_of_objects and clear_of_rectangles:
            return position
    raise RuntimeError("could not place all objects without overlap")


def add_object_assets_and_body(
    asset: ET.Element,
    worldbody: ET.Element,
    scene_object: SceneObject,
    source_dir: Path,
    physics_config: dict,
    rng: np.random.Generator,
    origin: Path,
    mesh_origin: Path,
) -> None:
    prefix = scene_object.body_name
    source_root = ET.parse(source_dir / "model.xml").getroot()
    source_meshes = source_root.find("asset").findall("mesh")

    ET.SubElement(
        asset,
        "texture",
        {
            "name": f"{prefix}_texture",
            "type": "2d",
            "file": xml_path(source_dir / "texture.png", origin),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": f"{prefix}_material",
            "texture": f"{prefix}_texture",
            "specular": "0.4",
            "shininess": "0.4",
        },
    )

    generated_mesh_names = []
    for index, source_mesh in enumerate(source_meshes):
        generated_name = f"{prefix}_mesh_{index}"
        generated_mesh_names.append(generated_name)
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": generated_name,
                "file": xml_path(source_dir / source_mesh.get("file"), mesh_origin),
                "scale": vector([scene_object.scale] * 3),
            },
        )

    quaternion = [
        math.cos(scene_object.yaw / 2.0),
        0.0,
        0.0,
        math.sin(scene_object.yaw / 2.0),
    ]
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": scene_object.body_name,
            "pos": vector(scene_object.position),
            "quat": vector(quaternion),
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"{prefix}_free"})

    size = np.asarray(scene_object.scaled_size)
    center = np.mean(np.asarray(scene_object.raw_bounds), axis=0) * scene_object.scale
    inertia = scene_object.mass / 12.0 * np.array(
        [
            size[1] ** 2 + size[2] ** 2,
            size[0] ** 2 + size[2] ** 2,
            size[0] ** 2 + size[1] ** 2,
        ]
    )
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": vector(center),
            "mass": f"{scene_object.mass:.9g}",
            "diaginertia": vector(np.maximum(inertia, 1e-8)),
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{prefix}_visual",
            "type": "mesh",
            "mesh": generated_mesh_names[0],
            "material": f"{prefix}_material",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": str(ROBOT_VISUAL_GROUP),
        },
    )

    variation = float(physics_config["relative_variation"])
    solref = jitter_vector(rng, physics_config["contact_solref"], variation)
    solimp = jitter_vector(rng, physics_config["contact_solimp"], variation)
    solimp[:2] = sorted(solimp[:2])
    for index, mesh_name in enumerate(generated_mesh_names):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_collision_{index}",
                "type": "mesh",
                "mesh": mesh_name,
                "density": "0",
                "group": str(COLLISION_GROUP),
                "friction": vector(scene_object.friction),
                "solref": vector(solref),
                "solimp": vector(solimp),
            },
        )


def make_scene_object(
    object_dir: Path,
    role: str,
    body_name: str,
    max_size: float,
    xy: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    yaw: float | None = None,
) -> SceneObject:
    bounds = mesh_bounds(object_dir)
    raw_size = np.ptp(bounds, axis=0)
    scale = max_size / float(np.max(raw_size))
    scaled_size = raw_size * scale
    z = -float(bounds[0, 2]) * scale + float(config["scene"]["spawn_clearance_m"])
    physics = config["scene"]["physics"]
    variation = float(physics["relative_variation"])
    estimated_mass = (
        float(np.prod(scaled_size))
        * float(physics["bounding_box_fill_fraction"])
        * float(physics["object_density_kg_m3"])
    )
    mass = np.clip(
        jitter(rng, estimated_mass, variation),
        float(physics["object_mass_kg"][0]),
        float(physics["object_mass_kg"][1]),
    )
    friction = jitter_vector(rng, physics["object_friction"], variation)
    return SceneObject(
        name=object_dir.name,
        role=role,
        body_name=body_name,
        source_dir=stored_path(object_dir),
        scale=float(scale),
        position=[float(xy[0]), float(xy[1]), z],
        yaw=float(rng.uniform(-math.pi, math.pi) if yaw is None else yaw),
        raw_bounds=bounds.tolist(),
        scaled_size=scaled_size.tolist(),
        mass=float(mass),
        friction=friction,
    )
