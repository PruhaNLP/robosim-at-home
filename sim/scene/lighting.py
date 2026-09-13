import math
import xml.etree.ElementTree as ET

import numpy as np

from core.common import uniform_range, vector

# ======Settings=========
# Lighting ranges are defined in scene/config.yaml.
# ======Settings=========


def kelvin_rgb(temperature_k: float) -> np.ndarray:
    temperature = temperature_k / 100.0
    if temperature <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = (
            0.0
            if temperature <= 19.0
            else 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
        )
    else:
        red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)
        blue = 255.0
    return np.clip([red, green, blue], 0.0, 255.0) / 255.0


def add_lights(
    worldbody: ET.Element,
    config: dict,
    table_center: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    light_config = config["scene"]["lights"]
    placements = light_config.get("positions_m")
    count = (
        len(placements)
        if placements
        else int(
            rng.integers(
                int(light_config["count"][0]),
                int(light_config["count"][1]) + 1,
            )
        )
    )
    lights = []
    target = np.array([table_center[0], table_center[1], 0.0], dtype=np.float64)

    for index in range(count):
        if placements:
            position = np.asarray(placements[index], dtype=np.float64)
        else:
            position = np.array(
                [
                    uniform_range(rng, light_config["position_x_m"]),
                    uniform_range(rng, light_config["position_y_m"]),
                    uniform_range(rng, light_config["position_z_m"]),
                ]
            )
        direction = target - position
        direction /= np.linalg.norm(direction)
        temperature = uniform_range(rng, light_config["color_temperature_k"])
        color = kelvin_rgb(temperature)
        sampled_intensity = uniform_range(rng, light_config["intensity"])
        intensity = (
            sampled_intensity
            * float(light_config["total_intensity_scale"])
            / count
        )
        ambient_level = uniform_range(rng, light_config["ambient"])
        specular_level = uniform_range(rng, light_config["specular"])
        directional = bool(
            rng.random() < float(light_config["directional_probability"])
        )
        cast_shadow = bool(
            rng.random() < float(light_config["cast_shadow_probability"])
        )
        attenuation = [
            1.0,
            uniform_range(rng, light_config["attenuation_linear"]),
            uniform_range(rng, light_config["attenuation_quadratic"]),
        ]
        light = ET.SubElement(
            worldbody,
            "light",
            {
                "name": f"random_light_{index}",
                "pos": vector(position),
                "dir": vector(direction),
                "directional": str(directional).lower(),
                "castshadow": str(cast_shadow).lower(),
                "diffuse": vector(color * intensity),
                "ambient": vector(color * ambient_level),
                "specular": vector(color * specular_level),
                "attenuation": vector(attenuation),
                "cutoff": f"{uniform_range(rng, light_config['cutoff_deg']):.9g}",
                "exponent": f"{uniform_range(rng, light_config['exponent']):.9g}",
            },
        )
        lights.append(
            {
                "name": light.get("name"),
                "position": position.tolist(),
                "direction": direction.tolist(),
                "temperature_k": temperature,
                "intensity": intensity,
                "sampled_intensity": sampled_intensity,
                "directional": directional,
                "cast_shadow": cast_shadow,
            }
        )
    return lights
