import json
import os
import sqlite3
import threading
from pathlib import Path

import yaml

# ======Settings=========
SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SIM_ROOT.parent
DATA_ROOT = Path(os.environ.get("ROBOSIM_DATA_DIR", REPO_ROOT / "data"))
if DATA_ROOT.is_absolute():
    DATA_ROOT = DATA_ROOT.resolve()
else:
    DATA_ROOT = (REPO_ROOT / DATA_ROOT).resolve()
PACKAGE_CONFIG = SIM_ROOT / "config.yaml"
DB_PATH = DATA_ROOT / "app.db"
DATASETS_DIR = DATA_ROOT / "datasets"
SCENES_DIR = DATA_ROOT / "scenes"
UI_SCENE_DIR = SCENES_DIR / "ui"
COLLECT_SCENE_DIR = SCENES_DIR / "collect"
GRPO_SCENE_DIR = SCENES_DIR / "grpo"
GENERATE_SCENE_DIR = SCENES_DIR / "generate"
TRAIN_DIR = DATA_ROOT / "train"
LEROBOT_DIR = DATA_ROOT / "lerobot"
LOGS_DIR = DATA_ROOT / "logs"
CACHE_DIR = DATA_ROOT / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
BENCH_DIR = DATA_ROOT / "bench"
EVAL_DIR = DATA_ROOT / "eval"
EVAL_VALSET_DIR = EVAL_DIR / "valset"
LLM_LOG_PATH = LOGS_DIR / "llm_calls.jsonl"
LEGACY_CONFIG_DIR = DATA_ROOT / "config"
POLICY_CAMERA_SLOTS = ("camera1", "camera2", "camera3", "camera4", "camera5")
SCENE_CAMERA_CHOICES = ("front", "wrist", "overview", "top", "left", "right")
DEFAULT_POLICY_MAP = {
    "camera1": "front",
    "camera2": "wrist",
    "camera3": "",
    "camera4": "",
    "camera5": "",
}
DEFAULT_COMPUTE = {
    "render": "auto",
    "model": "auto",
    "policy_map": dict(DEFAULT_POLICY_MAP),
    "policy_cameras": ["front", "wrist"],
}
DEFAULT_POLICY_MODE = "smolvla"
DEFAULT_TRAINING = {
    "num_workers": 4,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.95,
    "optimizer_eps": 1e-8,
    "optimizer_weight_decay": 1e-10,
    "grad_clip_norm": 10.0,
    "scheduler_warmup_steps": 1000,
    "scheduler_decay_steps": 1000,
    "scheduler_decay_lr": 2.5e-6,
    "use_amp": True,
    "vision_cache": True,
    "vision_cache_gb": 16,
    "frame_cache": True,
    "frame_cache_gb": 8,
    "warmup_fraction": 0.1,
    "final_lr_ratio": 0.025,
}
PACKAGE_FILES = {
    "scene": SIM_ROOT / "scene" / "config.yaml",
    "cameras": SIM_ROOT / "cameras" / "config.yaml",
    "language": SIM_ROOT / "language" / "config.yaml",
    "reward": SIM_ROOT / "rewards" / "config.yaml",
    "environment": SIM_ROOT / "core" / "environment.yaml",
}
ROOT_KEYS = ("seed", "compute", "training", "policy_mode", "paths")
SECTION_KEYS = ("scene", "cameras", "language", "reward", "environment")
LEGACY_FILES = {
    "scene": LEGACY_CONFIG_DIR / "scene.yaml",
    "cameras": LEGACY_CONFIG_DIR / "cameras.yaml",
    "language": LEGACY_CONFIG_DIR / "language.yaml",
    "reward": LEGACY_CONFIG_DIR / "rewards.yaml",
    "environment": LEGACY_CONFIG_DIR / "environment.yaml",
}
DATA_DIRS = (
    DATA_ROOT,
    DATASETS_DIR,
    UI_SCENE_DIR,
    COLLECT_SCENE_DIR,
    GRPO_SCENE_DIR,
    GENERATE_SCENE_DIR,
    TRAIN_DIR,
    LEROBOT_DIR,
    LOGS_DIR,
    HF_CACHE_DIR,
    BENCH_DIR,
    EVAL_DIR,
    EVAL_VALSET_DIR,
)
# ======Settings=========

_DB_LOCK = threading.Lock()
_READY = False


def stored_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside the repository: {path}") from error


def resolve_stored(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise ValueError(f"absolute paths are not allowed: {value}") from error
        return resolved
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside the repository: {value}") from error
    return resolved


def resolve_source(source: str) -> Path | str:
    raw = str(source or "").strip()
    if not raw:
        return raw
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(REPO_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return resolve_stored(candidate)
    return raw


def stored_source(source: str | Path) -> str:
    resolved = resolve_source(str(source))
    if isinstance(resolved, Path):
        return stored_path(resolved)
    return str(source)


def normalize_policy_map(value) -> dict[str, str]:
    mapping = {slot: "" for slot in POLICY_CAMERA_SLOTS}
    if isinstance(value, dict):
        for slot in POLICY_CAMERA_SLOTS:
            raw = value.get(slot, value.get(slot[6:]))
            name = str(raw or "").strip()
            if name.lower() in {"empty", "none", "-", "—"}:
                name = ""
            mapping[slot] = name
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if index >= len(POLICY_CAMERA_SLOTS):
                break
            mapping[POLICY_CAMERA_SLOTS[index]] = str(item or "").strip()
    return mapping


def policy_cameras_from_map(mapping) -> list[str]:
    mapping = normalize_policy_map(mapping)
    names: list[str] = []
    seen: set[str] = set()
    for slot in POLICY_CAMERA_SLOTS:
        name = mapping.get(slot) or ""
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def policy_rename_map(mapping) -> dict[str, str]:
    return {
        f"observation.images.{name}": f"observation.images.camera{index}"
        for index, name in enumerate(policy_cameras_from_map(mapping), start=1)
    }


def policy_cameras_from_config(config: dict | None) -> list[str]:
    compute = (config or {}).get("compute") or {}
    names = policy_cameras_from_map(compute.get("policy_map"))
    if names:
        return names
    return [
        str(item).strip()
        for item in (compute.get("policy_cameras") or [])
        if str(item).strip()
    ]


def apply_policy_map(compute: dict | None) -> dict:
    compute = dict(compute or {})
    mapping = normalize_policy_map(
        compute.get("policy_map")
        if compute.get("policy_map") not in (None, {})
        else compute.get("policy_cameras") or DEFAULT_POLICY_MAP
    )
    if not any(mapping.values()):
        mapping = dict(DEFAULT_POLICY_MAP)
    compute["policy_map"] = mapping
    compute["policy_cameras"] = policy_cameras_from_map(mapping)
    return compute


def _connect() -> sqlite3.Connection:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS settings ("
        "key TEXT PRIMARY KEY NOT NULL, "
        "value TEXT NOT NULL)"
    )
    return conn


def _db_get_unlocked(key: str):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return json.loads(row[0])


def _db_set_unlocked(key: str, value) -> None:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def _db_get(key: str):
    with _DB_LOCK:
        return _db_get_unlocked(key)


def _db_set(key: str, value) -> None:
    with _DB_LOCK:
        _db_set_unlocked(key, value)


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _legacy_relpath(value: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    root = REPO_ROOT.resolve()
    text = str(path)
    prefix = str(root)
    if text == prefix:
        return "."
    if text.startswith(prefix + os.sep):
        return Path(text[len(prefix) + 1 :]).as_posix()
    return None


def _rewrite_json_paths(payload):
    changed = False
    if isinstance(payload, dict):
        for key, value in list(payload.items()):
            if isinstance(value, str):
                rel = _legacy_relpath(value)
                if rel is not None:
                    payload[key] = rel
                    changed = True
            elif isinstance(value, (dict, list)):
                if _rewrite_json_paths(value):
                    changed = True
    elif isinstance(payload, list):
        for item in payload:
            if _rewrite_json_paths(item):
                changed = True
    return changed


def _migrate_legacy_yaml() -> None:
    root_file = LEGACY_CONFIG_DIR / "config.yaml"
    if root_file.is_file():
        user_root = _read_yaml(root_file)
        for key in ROOT_KEYS:
            if key in user_root and _db_get_unlocked(key) is None:
                _db_set_unlocked(key, user_root[key])
        root_file.unlink()
    for key, path in LEGACY_FILES.items():
        if path.is_file() and _db_get_unlocked(key) is None:
            _db_set_unlocked(key, _read_yaml(path))
        if path.is_file():
            path.unlink()
    if LEGACY_CONFIG_DIR.is_dir() and not any(LEGACY_CONFIG_DIR.iterdir()):
        LEGACY_CONFIG_DIR.rmdir()


def _migrate_stored_files() -> None:
    for meta in SCENES_DIR.glob("*/metadata.json"):
        payload = json.loads(meta.read_text())
        if _rewrite_json_paths(payload):
            meta.write_text(json.dumps(payload, indent=2) + "\n")
    for path in TRAIN_DIR.glob("**/train_config.json"):
        payload = json.loads(path.read_text())
        if _rewrite_json_paths(payload):
            path.write_text(json.dumps(payload, indent=2) + "\n")
    if LLM_LOG_PATH.is_file():
        lines = []
        changed = False
        for raw in LLM_LOG_PATH.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                lines.append(raw)
                continue
            if _rewrite_json_paths(event):
                changed = True
            lines.append(json.dumps(event, ensure_ascii=False))
        if changed:
            LLM_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_data_dirs() -> None:
    global _READY
    for path in DATA_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    with _DB_LOCK:
        if _READY:
            return
        _connect().close()
        _migrate_legacy_yaml()
        _migrate_stored_files()
        _READY = True


def normalize_policy_mode(value) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("act", "smolvla"):
        return raw
    return DEFAULT_POLICY_MODE


def _section(key: str) -> dict:
    user = _db_get(key)
    if user is not None:
        return user
    return _read_yaml(PACKAGE_FILES[key])


def load_config() -> dict:
    ensure_data_dirs()
    package_root = _read_yaml(PACKAGE_CONFIG)
    root = dict(package_root)
    for key in ROOT_KEYS:
        value = _db_get(key)
        if value is not None:
            root[key] = value
    paths = dict(package_root.get("paths") or {})
    paths.update(root.get("paths") or {})
    compute = apply_policy_map({**DEFAULT_COMPUTE, **(root.get("compute") or {})})
    training = dict(DEFAULT_TRAINING)
    training.update(root.get("training") or {})
    return {
        "seed": int(root["seed"]),
        "compute": compute,
        "training": training,
        "policy_mode": normalize_policy_mode(root.get("policy_mode")),
        "paths": paths,
        "scene": _section("scene"),
        "cameras": _section("cameras"),
        "language": _section("language"),
        "reward": _section("reward"),
        "environment": _section("environment"),
    }


def save_config(config: dict) -> None:
    ensure_data_dirs()
    _db_set("seed", int(config["seed"]))
    _db_set("compute", config.get("compute", dict(DEFAULT_COMPUTE)))
    _db_set("training", config.get("training", dict(DEFAULT_TRAINING)))
    _db_set("policy_mode", normalize_policy_mode(config.get("policy_mode")))
    _db_set("paths", config["paths"])
    for key in SECTION_KEYS:
        if key in config:
            _db_set(key, config[key])
