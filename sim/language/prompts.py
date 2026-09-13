import numpy as np

from core.common import humanize

# ======Settings=========
BASIC_COLORS = {
    "red": [0.85, 0.15, 0.15],
    "orange": [0.90, 0.45, 0.10],
    "yellow": [0.90, 0.85, 0.15],
    "green": [0.15, 0.70, 0.25],
    "blue": [0.15, 0.35, 0.85],
    "purple": [0.55, 0.20, 0.75],
    "pink": [0.90, 0.45, 0.65],
    "brown": [0.45, 0.25, 0.12],
    "gray": [0.50, 0.50, 0.50],
    "white": [0.90, 0.90, 0.90],
    "black": [0.10, 0.10, 0.10],
}
# ======Settings=========


def nearest_color(rgb: list[float]) -> str:
    color = np.asarray(rgb, dtype=np.float64)
    return min(
        BASIC_COLORS,
        key=lambda name: np.linalg.norm(
            color - np.asarray(BASIC_COLORS[name], dtype=np.float64)
        ),
    )


def generate_prompt(
    metadata: dict,
    config: dict,
    rng: np.random.Generator,
) -> str:
    prompt_config = config["language"]["prompts"]
    template = str(rng.choice(prompt_config["templates"]))
    destination_template = str(
        rng.choice(prompt_config["destination_templates"])
    )
    tray_color = nearest_color(metadata["appearance"]["tray_rgb"])
    target_aliases = config["language"]["target_labels"].get(
        metadata["target"],
        [humanize(metadata["target"]).lower()],
    )
    target = f"the {str(rng.choice(target_aliases))}"
    destination = destination_template.format(tray_color=tray_color)
    return template.format(
        target=target,
        destination=destination,
        tray_color=tray_color,
    )
