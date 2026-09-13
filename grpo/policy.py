"""SDE wrapper around the custom SmolVLA inference loop."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from core.config import resolve_source, stored_source

from .flow_sde import flow_scale_weights, recompute_step_logprob

# ======Settings=========
SDE_MODE_ALL = "all_steps"
SDE_MODE_ONE = "one_random_step"
# ======Settings=========


def resolve_checkpoint(source: str) -> str:
    resolved = resolve_source(source)
    path = resolved if isinstance(resolved, Path) else Path(source)
    if path.is_dir():
        if (path / "config.json").is_file() or (path / "model.safetensors").is_file():
            return stored_source(path)
        nested = path / "pretrained_model"
        if (nested / "config.json").is_file() or (nested / "model.safetensors").is_file():
            return stored_source(nested)
        if path.name == "last" and path.parent.is_dir():
            numbered = sorted(
                item
                for item in path.parent.iterdir()
                if item.name.isdigit()
                and (
                    (item / "pretrained_model" / "model.safetensors").is_file()
                    or (item / "pretrained_model" / "config.json").is_file()
                )
            )
            if numbered:
                return stored_source(numbered[-1] / "pretrained_model")
    return source


class SdeRollout:
    __slots__ = (
        "action",
        "traj",
        "taus",
        "dtau",
        "step_logprobs",
        "step_mask",
        "step_weights",
    )

    def __init__(
        self,
        action: torch.Tensor,
        traj: torch.Tensor,
        taus: torch.Tensor,
        dtau: float,
        step_logprobs: torch.Tensor,
        step_mask: torch.Tensor,
        step_weights: torch.Tensor,
    ) -> None:
        self.action = action
        self.traj = traj
        self.taus = taus
        self.dtau = dtau
        self.step_logprobs = step_logprobs
        self.step_mask = step_mask
        self.step_weights = step_weights


def _obs_to_device(obs: dict, device: torch.device) -> dict:
    out = {"n_real": int(obs["n_real"])}
    for key in ("images", "tokens", "mask", "state"):
        value = obs[key]
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


class FlowSdePolicy(nn.Module):
    kind = "smolvla"
    def __init__(
        self,
        engine,
        num_steps: int,
        noise_level: float,
        freeze_vlm: bool,
        sde_mode: str,
        flow_scale: dict,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.policy = engine.policy
        self.num_steps = max(1, int(num_steps))
        self.noise_level = float(noise_level)
        self.freeze_vlm = bool(freeze_vlm)
        self.sde_mode = sde_mode if sde_mode in (SDE_MODE_ALL, SDE_MODE_ONE) else SDE_MODE_ONE
        self.flow_scale = dict(flow_scale)
        self._set_trainable()
        if hasattr(self.policy, "reset_runtime"):
            self.policy.reset_runtime()
        self.policy.to(dtype=torch.float32)
        self.engine._compute_dtype = torch.float32

    def _set_trainable(self) -> None:
        if self.policy is None:
            raise RuntimeError("engine is not loaded")
        for name, param in self.policy.named_parameters():
            if self.freeze_vlm and "vlm_with_expert.vlm" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def _choose_sde_steps(self, weights: torch.Tensor) -> set[int]:
        if self.sde_mode == SDE_MODE_ALL:
            return set(range(self.num_steps))
        probs = weights.detach().float().clamp(min=1e-8)
        probs = probs / probs.sum()
        return {int(torch.multinomial(probs, 1).item())}

    def _taus(self, device: torch.device) -> tuple[torch.Tensor, float]:
        dtau = -1.0 / self.num_steps
        taus = torch.tensor(
            [1.0 + dtau * step for step in range(self.num_steps)],
            device=device,
            dtype=torch.float32,
        )
        return taus, dtau

    @torch.inference_mode()
    def rollout(self, obs: dict) -> SdeRollout:
        obs = _obs_to_device(obs, self.engine.device)
        device = self.engine.device
        taus, dtau = self._taus(device)
        weights = flow_scale_weights(
            taus,
            enabled=bool(self.flow_scale.get("enabled", True)),
            power=float(self.flow_scale.get("power", 0.5)),
            uniform_mix=float(self.flow_scale.get("uniform_mix", 0.5)),
            min_weight=float(self.flow_scale.get("min_weight", 0.5)),
            max_weight=float(self.flow_scale.get("max_weight", 1.5)),
        )
        action, traj, logps, mask, dtau = self.policy.sample_actions_sde(
            obs["images"],
            obs["n_real"],
            obs["tokens"],
            obs["mask"],
            obs["state"],
            self.num_steps,
            self.noise_level,
            self._choose_sde_steps(weights),
        )
        return SdeRollout(
            action=action,
            traj=traj,
            taus=taus,
            dtau=dtau,
            step_logprobs=logps,
            step_mask=mask,
            step_weights=weights,
        )

    def recompute_logprobs(self, obs: dict, rollout: SdeRollout) -> torch.Tensor:
        obs = _obs_to_device(obs, rollout.traj.device)
        cache_k, cache_v, expert_k, expert_v, suffix = self.policy.prefix_caches(
            obs["images"],
            obs["n_real"],
            obs["tokens"],
            obs["mask"],
            obs["state"],
            bind_static=False,
            vlm_no_grad=self.freeze_vlm,
        )
        self_mask, cross_mask, self_cos, self_sin, cross_cos, cross_sin = suffix
        batch = int(rollout.traj.shape[1])
        step_mask = rollout.step_mask
        if step_mask.ndim == 1:
            step_mask = step_mask.view(-1, 1).expand(-1, batch)
        rows = []
        for step in range(self.num_steps):
            if not bool(step_mask[step].any()):
                rows.append(torch.zeros(batch, device=rollout.traj.device))
                continue
            x = rollout.traj[step]
            x_next = rollout.traj[step + 1]
            tau = torch.full((x.shape[0],), float(rollout.taus[step]), device=x.device)
            velocity = self.policy._denoise_step(
                x,
                tau,
                cache_k,
                cache_v,
                expert_k,
                expert_v,
                self_mask=self_mask,
                cross_mask=cross_mask,
                self_cos=self_cos,
                self_sin=self_sin,
                cross_cos=cross_cos,
                cross_sin=cross_sin,
            )
            rows.append(
                recompute_step_logprob(
                    x,
                    x_next,
                    velocity,
                    tau,
                    rollout.dtau,
                    self.noise_level,
                )
            )
        return torch.stack(rows, dim=0)


def load_engine(source: str, device: torch.device, on_log=None, cameras=None):
    from model.act import is_act_checkpoint
    from model.inference import SmolVLAEngine

    resolved = resolve_checkpoint(source)
    if is_act_checkpoint(resolved):
        raise RuntimeError("GRPO does not support ACT checkpoints.")
    engine = SmolVLAEngine(device=str(device))
    engine.load(source=resolved, on_log=on_log, cameras=cameras, warmup=False)
    return engine, resolved
