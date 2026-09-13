"""Flow-SDE: ODE→SDE conversion for SmolVLA (π_RL §5.1 / Flow-GRPO).

SmolVLA denoises with Euler ODE, τ: 1 → 0:
    A_{k+1} = A_k + dτ · v_θ(A_k, τ_k),  dτ = -1/N

The equivalent reverse-time SDE (same marginals, π_RL Eq. 8):
    dA^τ = [v_θ + σ_τ²/(2τ) · (A + (1-τ) v_θ)] dτ + σ_τ dw_τ
    σ_τ = a · √(τ / (1-τ))

Euler–Maruyama (dτ < 0):
    μ_τ = A + dτ · [v + σ²/(2τ) · (A + (1-τ) v)]
    A_next ~ N(μ_τ, (σ √|dτ|)² I)

Each transition is Gaussian, so log π(A_{k+1}|A_k) is closed-form.
GRPO holds (A_k, A_{k+1}) fixed and recomputes μ_τ under new θ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ======Settings=========
EPS = 1e-5
STD_FLOOR = 1e-3
TAU_SIGMA_MAX = 0.9
LOGPROB_ACTION_DIM = 6
# ======Settings=========


def _joint_logprob(x_next: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    per_elem = -0.5 * ((x_next - mean) ** 2 / var + torch.log(2.0 * math.pi * var))
    joint = per_elem[..., :LOGPROB_ACTION_DIM]
    return joint.flatten(1).sum(dim=1)


@dataclass
class SdeStepResult:
    x_next: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    log_prob: torch.Tensor


def sigma_tau(tau: torch.Tensor, noise_level: float) -> torch.Tensor:
    tau_c = tau.clamp(min=EPS, max=TAU_SIGMA_MAX)
    return noise_level * torch.sqrt(tau_c / (1.0 - tau_c))


def sde_step(
    x: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    noise_level: float,
    noise: torch.Tensor | None = None,
) -> SdeStepResult:
    if tau.ndim == 0:
        tau = tau.expand(x.shape[0])
    tau_view = tau.view(-1, *([1] * (x.ndim - 1)))
    sigma = sigma_tau(tau_view, noise_level)
    tau_safe = tau_view.clamp(min=EPS)
    drift = v + (sigma**2) / (2.0 * tau_safe) * (x + (1.0 - tau_view) * v)
    mean = x + dtau * drift
    std = (sigma * math.sqrt(abs(dtau))).clamp(min=STD_FLOOR)
    if noise is None:
        noise = torch.randn_like(x)
    x_next = mean + std * noise
    var = std**2
    log_prob = _joint_logprob(x_next, mean, var)
    return SdeStepResult(
        x_next=x_next,
        mean=mean,
        log_std=torch.log(std).expand_as(x),
        log_prob=log_prob,
    )


def recompute_step_logprob(
    x: torch.Tensor,
    x_next: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    dtau: float,
    noise_level: float,
) -> torch.Tensor:
    if tau.ndim == 0:
        tau = tau.expand(x.shape[0])
    tau_view = tau.view(-1, *([1] * (x.ndim - 1)))
    sigma = sigma_tau(tau_view, noise_level)
    tau_safe = tau_view.clamp(min=EPS)
    drift = v + (sigma**2) / (2.0 * tau_safe) * (x + (1.0 - tau_view) * v)
    mean = x + dtau * drift
    std = (sigma * math.sqrt(abs(dtau))).clamp(min=STD_FLOOR)
    var = std**2
    return _joint_logprob(x_next, mean, var)


def flow_scale_weights(
    taus: torch.Tensor,
    enabled: bool,
    power: float,
    uniform_mix: float,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    if not enabled or taus.numel() == 0:
        return torch.ones_like(taus)
    mix = min(max(float(uniform_mix), 0.0), 1.0)
    raw = mix + (1.0 - mix) * taus.clamp(min=0.0, max=1.0).pow(float(power))
    low = raw.min()
    high = raw.max()
    if float(high - low) < 1e-8:
        scaled = torch.full_like(raw, 0.5 * (min_weight + max_weight))
    else:
        scaled = min_weight + (raw - low) / (high - low) * (max_weight - min_weight)
    return scaled
