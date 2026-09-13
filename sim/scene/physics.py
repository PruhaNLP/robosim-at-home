import xml.etree.ElementTree as ET

import numpy as np

from core.common import jitter, jitter_vector, vector

# ======Settings=========
# Physical ranges are defined in scene/config.yaml.
# ======Settings=========


def randomize_robot_physics(
    root: ET.Element, rng: np.random.Generator, variation: float
) -> dict:
    body_scales = {}
    for body in root.findall(".//body"):
        inertial = body.find("inertial")
        if inertial is None:
            continue
        factor = float(rng.uniform(1.0 - variation, 1.0 + variation))
        inertial.set("mass", f"{float(inertial.get('mass')) * factor:.9g}")
        inertia = [float(value) * factor for value in inertial.get("diaginertia").split()]
        inertial.set("diaginertia", vector(inertia))
        body_scales[body.get("name", "unnamed")] = factor

    for joint in root.findall(".//default/joint"):
        for attribute in ("frictionloss", "armature", "damping"):
            if joint.get(attribute) is not None:
                joint.set(attribute, f"{jitter(rng, float(joint.get(attribute)), variation):.9g}")

    for position in root.findall(".//default/position"):
        for attribute in ("kp", "dampratio"):
            if position.get(attribute) is not None:
                position.set(
                    attribute,
                    f"{jitter(rng, float(position.get(attribute)), variation):.9g}",
                )
        if position.get("forcerange") is not None:
            force_range = [float(value) for value in position.get("forcerange").split()]
            factor = float(rng.uniform(1.0 - variation, 1.0 + variation))
            position.set("forcerange", vector([value * factor for value in force_range]))

    for geom in root.findall(".//geom"):
        if geom.get("friction") is not None:
            friction = [float(value) for value in geom.get("friction").split()]
            geom.set("friction", vector(jitter_vector(rng, friction, variation)))

    return body_scales
