import ctypes
import ctypes.util
import os
import re
import subprocess
import threading
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_int, c_void_p
from pathlib import Path

# ======Settings=========
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_EXTENSIONS = 0x3055
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054
SOFTWARE_EXTENSION = "EGL_MESA_device_software"
PCI_IDS_PATHS = (
    Path("/usr/share/misc/pci.ids"),
    Path("/usr/share/hwdata/pci.ids"),
)
PCI_VENDOR_NAME = {
    "10de": "NVIDIA",
    "1002": "AMD",
    "8086": "Intel",
}
PCI_BASE_DISPLAY = 0x03
PCI_SUBCLASS_VGA = 0x00
PCI_SUBCLASS_OTHER_DISPLAY = 0x80
# ======Settings=========

_DEVICE_CACHE = None
_MODEL_CACHE = None
_DEVICE_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()


def _load_egl():
    library = ctypes.util.find_library("EGL") or "libEGL.so.1"
    egl = ctypes.CDLL(library)
    egl.eglGetProcAddress.restype = c_void_p
    egl.eglGetProcAddress.argtypes = [c_char_p]
    egl.eglInitialize.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
    egl.eglInitialize.restype = c_int
    egl.eglTerminate.argtypes = [c_void_p]
    egl.eglTerminate.restype = c_int
    egl.eglQueryString.argtypes = [c_void_p, c_int]
    egl.eglQueryString.restype = c_char_p
    return egl


def _egl_proc(egl, name, restype, argtypes):
    address = egl.eglGetProcAddress(name.encode("ascii"))
    if not address:
        raise RuntimeError(f"EGL extension {name} is missing")
    return CFUNCTYPE(restype, *argtypes)(address)


def _norm_pci(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.lower().split(":")
    if len(parts) >= 3 and len(parts[0]) > 4:
        parts[0] = parts[0][-4:]
    return ":".join(parts)


def _decode(value) -> str | None:
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _nvidia_cards() -> list[dict]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    cards = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        cards.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_mb": int(float(parts[2])) if len(parts) >= 3 else None,
                "pci": _norm_pci(parts[3]) if len(parts) >= 4 else None,
            }
        )
    return cards


def _hex_id(value: str | None) -> str | None:
    raw = str(value or "").strip().lower().replace("0x", "")
    if not raw:
        return None
    return raw


def _sysfs_pci_ids(pci_path: Path) -> tuple[str | None, str | None]:
    try:
        vendor = _hex_id((pci_path / "vendor").read_text())
        device = _hex_id((pci_path / "device").read_text())
    except OSError:
        return None, None
    return vendor, device


def _pretty_pci_label(name: str) -> str:
    name = name.strip()
    bracket = re.findall(r"\[([^\]]+)\]", name)
    return bracket[-1] if bracket else name


def _pci_ids_lookup(vendor: str, device: str | None = None) -> tuple[str | None, str | None]:
    vendor = vendor.lower()
    device = device.lower() if device else None
    for path in PCI_IDS_PATHS:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        vendor_name = None
        in_vendor = False
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if not line.startswith("\t"):
                in_vendor = line.lower().startswith(vendor + " ")
                if in_vendor:
                    vendor_name = _pretty_pci_label(line.split(" ", 1)[1])
                continue
            if (
                device
                and in_vendor
                and line.startswith("\t")
                and not line.startswith("\t\t")
            ):
                ident, _, rest = line.strip().partition(" ")
                if ident.lower() == device:
                    return vendor_name, _pretty_pci_label(rest)
        if vendor_name is not None and device is None:
            return vendor_name, None
        if vendor_name is not None:
            return vendor_name, None
    return None, None


def _pci_class_code(pci_path: Path) -> int | None:
    try:
        return int((pci_path / "class").read_text().strip(), 16)
    except (OSError, ValueError):
        return None


def _boot_vga(pci_path: Path) -> bool:
    try:
        return (pci_path / "boot_vga").read_text().strip() == "1"
    except OSError:
        return False


def _drm_connectors(card_node: str) -> list[str]:
    prefix = f"{card_node}-"
    root = Path("/sys/class/drm")
    if not root.is_dir():
        return []
    names = []
    for path in root.iterdir():
        if path.name.startswith(prefix):
            names.append(path.name[len(prefix) :])
    return sorted(names)


def _is_display_gpu(pci_class: int | None, connectors: list[str]) -> bool:
    if connectors:
        return True
    if pci_class is None:
        return False
    base = (pci_class >> 16) & 0xFF
    subclass = (pci_class >> 8) & 0xFF
    return base == PCI_BASE_DISPLAY and subclass in (
        PCI_SUBCLASS_VGA,
        PCI_SUBCLASS_OTHER_DISPLAY,
    )


def _pci_name(pci_path: Path) -> str | None:
    slot = pci_path.name
    if slot.startswith("0000:"):
        slot = slot[5:]
    try:
        output = subprocess.check_output(
            ["lspci", "-s", slot, "-mm"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    fields = [field.strip().strip('"') for field in output.split('"') if field.strip()]
    if len(fields) >= 4:
        name = f"{fields[2]} {fields[3]}".strip()
    else:
        name = output
    return _pretty_pci_label(name)


def _drm_info(drm_file: str) -> dict:
    node = Path(drm_file).name
    sys_device = Path("/sys/class/drm") / node / "device"
    info = {
        "drm": drm_file,
        "name": node,
        "pci": None,
        "vendor": None,
        "vendor_id": None,
        "device_id": None,
        "display": False,
        "boot_vga": False,
    }
    if not sys_device.exists():
        return info
    try:
        pci_path = sys_device.resolve()
        info["pci"] = _norm_pci(pci_path.name)
        vendor, device = _sysfs_pci_ids(pci_path)
        info["vendor_id"] = vendor
        info["device_id"] = device
        pci_ids_vendor, pci_ids_device = (
            _pci_ids_lookup(vendor, device) if vendor else (None, None)
        )
        info["vendor"] = (
            pci_ids_vendor
            or PCI_VENDOR_NAME.get(vendor or "")
        )
        info["name"] = (
            pci_ids_device
            or _pci_name(pci_path)
            or (f"{info['vendor']} GPU" if info["vendor"] else node)
        )
        connectors = _drm_connectors(node)
        info["boot_vga"] = _boot_vga(pci_path)
        info["display"] = _is_display_gpu(_pci_class_code(pci_path), connectors)
    except OSError:
        pass
    return info


def _enumerate_egl() -> list[dict]:
    egl = _load_egl()
    query_devices = _egl_proc(
        egl, "eglQueryDevicesEXT", c_int, [c_int, POINTER(c_void_p), POINTER(c_int)]
    )
    query_device_string = _egl_proc(
        egl, "eglQueryDeviceStringEXT", c_char_p, [c_void_p, c_int]
    )
    count = c_int()
    if not query_devices(0, None, ctypes.byref(count)) or count.value <= 0:
        return _fallback_render_devices()
    handles = (c_void_p * count.value)()
    got = c_int()
    query_devices(count.value, handles, ctypes.byref(got))
    nvidia = _nvidia_cards()
    devices = []
    for index, handle in enumerate(handles):
        extensions = _decode(query_device_string(handle, EGL_EXTENSIONS)) or ""
        drm_file = _decode(query_device_string(handle, EGL_DRM_DEVICE_FILE_EXT))
        software = SOFTWARE_EXTENSION in extensions
        drm = _drm_info(drm_file) if drm_file else {}
        pci = drm.get("pci")
        nvidia_match = next(
            (card for card in nvidia if card["pci"] and card["pci"] == pci),
            None,
        )
        if software:
            continue
        device_id = f"drm:{drm_file}" if drm_file else f"egl:{index}"
        vendor = "NVIDIA" if nvidia_match else (drm.get("vendor") or "GPU")
        display = bool(drm.get("display"))
        if nvidia_match:
            label = nvidia_match["name"]
            memory = nvidia_match["memory_mb"]
            detail = f"NVIDIA · EGL {index}"
            if memory:
                detail = f"{detail} · {memory} MB"
        else:
            label = drm.get("name") or f"GPU {index}"
            detail = f"{vendor} · EGL {index}"
        if not display:
            detail = f"{detail} · compute"
        devices.append(
            {
                "id": device_id,
                "kind": "gpu",
                "label": label,
                "detail": detail,
                "egl_index": index,
                "drm": drm_file,
                "vendor": vendor,
                "display": display,
                "boot_vga": bool(drm.get("boot_vga")),
            }
        )
    return devices or _fallback_render_devices()


def _fallback_render_devices() -> list[dict]:
    devices = []
    for card in _nvidia_cards():
        drm = {}
        if card.get("pci"):
            for path in Path("/sys/class/drm").glob("card[0-9]"):
                info = _drm_info(f"/dev/dri/{path.name}")
                if info.get("pci") == card["pci"]:
                    drm = info
                    break
        if not drm.get("display"):
            continue
        devices.append(
            {
                "id": f"cuda:{card['index']}",
                "kind": "gpu",
                "label": card["name"],
                "detail": f"NVIDIA · cuda:{card['index']}",
                "egl_index": card["index"],
                "drm": drm.get("drm"),
                "vendor": "NVIDIA",
                "display": True,
                "boot_vga": bool(drm.get("boot_vga")),
            }
        )
    return devices


def list_model_devices() -> list[dict]:
    global _MODEL_CACHE
    with _MODEL_LOCK:
        cached = _MODEL_CACHE
    if cached is not None:
        return list(cached)
    cards = []
    for card in _nvidia_cards():
        memory = (
            f", {card['memory_mb']} MB" if card.get("memory_mb") is not None else ""
        )
        cards.append(
            {
                "id": f"cuda:{card['index']}",
                "kind": "gpu",
                "label": f"CUDA {card['index']}",
                "detail": f"{card['name']}{memory}",
            }
        )
    with _MODEL_LOCK:
        if _MODEL_CACHE is None:
            _MODEL_CACHE = cards
        return list(_MODEL_CACHE)


def cached_model_devices() -> list[dict] | None:
    with _MODEL_LOCK:
        cached = _MODEL_CACHE
    if cached is None:
        return None
    return list(cached)


def resolve_model_device(value: str | None) -> str:
    raw = str(value or "auto").strip().lower()
    gpu_ids = [device["id"] for device in list_model_devices()]
    if raw in ("auto", "cuda"):
        return gpu_ids[0] if gpu_ids else "cpu"
    if raw == "cpu":
        return "cpu"
    if raw in gpu_ids:
        return raw
    if raw.startswith("cuda") and gpu_ids:
        return gpu_ids[0]
    return "cpu"


def list_render_devices() -> list[dict]:
    global _DEVICE_CACHE
    with _DEVICE_LOCK:
        cached = _DEVICE_CACHE
    if cached is not None:
        return list(cached)
    devices = _enumerate_egl()
    if not any(device["id"] == "cpu" for device in devices):
        devices.append(
            {
                "id": "cpu",
                "kind": "cpu",
                "label": "CPU",
                "detail": "software fallback",
                "egl_index": None,
                "drm": None,
                "display": False,
                "boot_vga": False,
            }
        )
    with _DEVICE_LOCK:
        if _DEVICE_CACHE is None:
            _DEVICE_CACHE = devices
        return list(_DEVICE_CACHE)


def cached_render_devices() -> list[dict] | None:
    with _DEVICE_LOCK:
        cached = _DEVICE_CACHE
    if cached is None:
        return None
    return list(cached)


def resolve_render_device(compute: dict) -> dict:
    devices = list_render_devices()
    if not devices:
        raise RuntimeError("no render devices found")
    selected = compute.get("render", "auto")
    if selected is None or selected == "auto":
        display = [
            device
            for device in devices
            if device["kind"] == "gpu" and device.get("display")
        ]
        boot = next((device for device in display if device.get("boot_vga")), None)
        if boot is not None:
            return boot
        if display:
            return display[0]
        gpus = [
            device
            for device in devices
            if device["kind"] == "gpu" and device.get("egl_index") is not None
        ]
        if gpus:
            return gpus[0]
        return next((device for device in devices if device["id"] == "cpu"), devices[0])
    if isinstance(selected, int) or (
        isinstance(selected, str) and selected.isdigit()
    ):
        index = int(selected)
        for device in devices:
            if device["egl_index"] == index:
                return device
        raise RuntimeError(
            f"EGL device {index} is not available; have "
            f"{[device['id'] for device in devices]}"
        )
    for device in devices:
        if device["id"] == selected:
            return device
    raise RuntimeError(
        f"render device {selected!r} is not available; have "
        f"{[device['id'] for device in devices]}"
    )


def apply_render_device(device: dict, environ: dict | None = None) -> None:
    target = os.environ if environ is None else environ
    if device.get("egl_index") is None:
        raise RuntimeError(f"device {device['id']} has no EGL index")
    target["MUJOCO_GL"] = "egl"
    target["MUJOCO_EGL_DEVICE_ID"] = str(device["egl_index"])
    target["ROBOSIM_RENDER_EGL_DEVICE_ID"] = str(device["egl_index"])
