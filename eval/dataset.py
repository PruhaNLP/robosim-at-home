from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.config import EVAL_VALSET_DIR

# ======Settings=========
MANIFEST_NAME = "manifest.json"
# ======Settings=========


def valset_root() -> Path:
    EVAL_VALSET_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_VALSET_DIR


def scene_dir(index: int) -> Path:
    return valset_root() / f"{int(index):04d}"


def public_scene(index: int, metadata: dict) -> dict:
    return {
        "index": int(index),
        "seed": int(metadata.get("seed") or 0),
        "instruction": str(metadata.get("instruction") or ""),
        "target": str(metadata.get("target") or ""),
        "distractors": list(metadata.get("distractors") or []),
        "cameras": list(metadata.get("cameras") or []),
        "room": metadata.get("room"),
    }


def empty_manifest() -> dict:
    return {"size": 0, "seed": 0, "scenes": []}


def load_manifest() -> dict:
    path = valset_root() / MANIFEST_NAME
    if not path.is_file():
        return empty_manifest()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty_manifest()
    if not isinstance(payload, dict):
        return empty_manifest()
    scenes = []
    for item in payload.get("scenes") or []:
        if not isinstance(item, dict):
            continue
        scenes.append(
            {
                "index": int(item.get("index") or len(scenes)),
                "seed": int(item.get("seed") or 0),
                "instruction": str(item.get("instruction") or ""),
                "target": str(item.get("target") or ""),
                "distractors": list(item.get("distractors") or []),
                "cameras": list(item.get("cameras") or []),
                "room": item.get("room"),
            }
        )
    return {
        "size": int(payload.get("size") or len(scenes)),
        "seed": int(payload.get("seed") or 0),
        "scenes": scenes,
    }


def save_manifest(payload: dict) -> dict:
    data = {
        "size": int(payload.get("size") or len(payload.get("scenes") or [])),
        "seed": int(payload.get("seed") or 0),
        "scenes": list(payload.get("scenes") or []),
    }
    path = valset_root() / MANIFEST_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return data


def write_scene(index: int, seed: int) -> dict:
    dest = scene_dir(index)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    from scene.generate import generate

    metadata = generate(seed_override=int(seed), output_dir_override=dest)
    return public_scene(index, metadata)


def next_seed(manifest: dict) -> int:
    used = {int(item.get("seed") or 0) for item in manifest.get("scenes") or []}
    seed = int(manifest.get("seed") or 0) + int(manifest.get("size") or 0)
    while seed in used:
        seed += 1
    return seed


def build_valset(size: int, seed0: int, should_stop, on_progress=None, on_log=None) -> dict:
    size = max(1, int(size))
    seed0 = int(seed0)
    root = valset_root()
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name != MANIFEST_NAME:
            child.unlink()
    scenes = []
    for index in range(size):
        if should_stop():
            break
        seed = seed0 + index
        if on_progress is not None:
            on_progress("build", index, size, f"Scene {index + 1}/{size} · generate")
        if on_log is not None:
            on_log(f"Generate val scene {index + 1}/{size} seed {seed}.")
        scenes.append(write_scene(index, seed))
    manifest = save_manifest({"size": len(scenes), "seed": seed0, "scenes": scenes})
    if on_progress is not None:
        on_progress("done", len(scenes), len(scenes), f"{len(scenes)} scenes")
    return manifest


def reroll_scene(index: int, seed: int | None = None) -> dict:
    manifest = load_manifest()
    scenes = list(manifest.get("scenes") or [])
    if index < 0 or index >= len(scenes):
        raise RuntimeError(f"no val scene {index}")
    next_value = int(seed) if seed is not None else next_seed(manifest)
    scenes[index] = write_scene(index, next_value)
    return save_manifest({**manifest, "scenes": scenes})
