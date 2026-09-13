import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.common import xml_path

# ======Settings=========
SKYBOX_GRID_SIZE = "3 4"
SKYBOX_GRID_LAYOUT = ".U..LFRB.D.."
# ======Settings=========


def available_rooms(rooms_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in rooms_dir.iterdir()
        if path.is_dir() and (path / "skybox.png").is_file()
    )


def choose_room(rooms_dir: Path, rng: np.random.Generator) -> Path:
    rooms = available_rooms(rooms_dir)
    if not rooms:
        raise FileNotFoundError(f"no prepared skyboxes in {rooms_dir}")
    return rooms[int(rng.integers(len(rooms)))]


def add_room_skybox(asset: ET.Element, room_dir: Path, origin: Path) -> None:
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "room_skybox",
            "type": "skybox",
            "file": xml_path(room_dir / "skybox.png", origin),
            "gridsize": SKYBOX_GRID_SIZE,
            "gridlayout": SKYBOX_GRID_LAYOUT,
        },
    )
