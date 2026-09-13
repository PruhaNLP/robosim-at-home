from __future__ import annotations

import torch

# ======Settings=========
BF16_MIN_MAJOR = 8
# ======Settings=========


def resolve_amp(device: torch.device, use_amp: bool) -> tuple[bool, torch.dtype, bool]:
    """Return (enabled, dtype, needs_scaler) for the current GPU.

    V100 (CC 7.0) cannot run BF16 compute. Official LeRobot AMP maps to BF16
    and is therefore disabled in the UI. This loop uses FP16 + GradScaler.
    """
    if not use_amp or device.type != "cuda" or not torch.cuda.is_available():
        return False, torch.float32, False
    major, _minor = torch.cuda.get_device_capability(device.index or 0)
    if major >= BF16_MIN_MAJOR and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def make_scaler(enabled: bool, device: torch.device):
    if not enabled:
        return None
    return torch.amp.GradScaler(device.type, enabled=True)
