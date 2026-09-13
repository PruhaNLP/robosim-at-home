"""PPO-style GRPO on Flow-SDE step log-probs (Flow-GRPO Eq. 5 / π_RL Eq. 5).

Ratio is per denoising step. Element log-probs are SUMmed inside a step
(chunk × action_dim). Do not divide by n_elem — that collapses the gradient
by ~1/1600 (SmolVLA_RL run5).

Clip is a trust region vs θ_old (rollout policy).
KL (k3) is an optional anchor to the frozen SFT reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# ======Settings=========
LOG_RATIO_CLAMP = 10.0
ADV_STD_FLOOR = 1e-6
# ======Settings=========


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    clip_fraction: torch.Tensor
    mean_ratio: torch.Tensor
    mean_advantage: torch.Tensor
    drift: torch.Tensor
    loss_mag: torch.Tensor


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    std = rewards.std(unbiased=False)
    if float(std) < ADV_STD_FLOOR:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + ADV_STD_FLOOR)


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    logp_ref: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
) -> GRPOLossOutput:
    log_ratio = (logp_new - logp_old).reshape(-1).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    ratio = torch.exp(log_ratio)
    adv = advantages.reshape(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per = -torch.min(unclipped, clipped)
    if step_weights is not None:
        weights = step_weights.reshape(-1).to(device=per.device, dtype=per.dtype)
        weights = weights / weights.mean().clamp(min=1e-8)
        policy_loss = (per * weights).mean()
    else:
        policy_loss = per.mean()

    if logp_ref is not None and kl_coef > 0.0:
        diff = logp_ref - logp_new
        kl_est = (torch.exp(diff.clamp(max=LOG_RATIO_CLAMP)) - diff - 1.0).mean()
        kl_loss = kl_coef * kl_est
    else:
        kl_loss = torch.zeros((), device=logp_new.device, dtype=logp_new.dtype)

    total = policy_loss + kl_loss
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        mean_ratio = ratio.mean()
        mean_adv = advantages.mean()
        loss_mag = per.detach().abs().mean()
        if logp_ref is not None:
            drift = (logp_ref - logp_new).abs().mean()
        else:
            drift = torch.zeros((), device=logp_new.device)
    return GRPOLossOutput(
        loss=total,
        policy_loss=policy_loss.detach(),
        kl_loss=kl_loss.detach(),
        clip_fraction=clip_frac,
        mean_ratio=mean_ratio,
        mean_advantage=mean_adv,
        drift=drift,
        loss_mag=loss_mag,
    )
