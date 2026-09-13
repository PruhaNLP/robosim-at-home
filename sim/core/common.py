import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ======Settings=========
# Shared helpers have no runtime settings.
# ======Settings=========


@dataclass
class SceneObject:
    name: str
    role: str
    body_name: str
    source_dir: str
    scale: float
    position: list[float]
    yaw: float
    raw_bounds: list[list[float]]
    scaled_size: list[float]
    mass: float
    friction: list[float]


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {value}")
    return (base / path).resolve()


def xml_path(path: Path, origin: Path) -> str:
    return Path(os.path.relpath(path.resolve(), origin.resolve())).as_posix()


def vector(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def jitter(rng: np.random.Generator, value: float, variation: float) -> float:
    return float(value * rng.uniform(1.0 - variation, 1.0 + variation))


def jitter_vector(
    rng: np.random.Generator, values: list[float], variation: float
) -> list[float]:
    return [jitter(rng, float(value), variation) for value in values]


def uniform_range(rng: np.random.Generator, values: list[float]) -> float:
    return float(rng.uniform(float(values[0]), float(values[1])))


def humanize(name: str) -> str:
    return " ".join(name.replace("_", " ").split())
