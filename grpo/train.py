"""Online Flow-SDE GRPO loop: scene → group rollouts → GRPO update."""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from core.config import policy_cameras_from_config, policy_rename_map
from scene.generate import generate

from .loss import group_advantages, grpo_loss
from .policy import FlowSdePolicy, SdeRollout, load_engine
from .rollout import EpisodeRecord, collect_scene_wave, encode_jpeg

# ======Settings=========
EPISODE_EQUAL_WEIGHT = True
DEFAULT_KL_COEF = 0.18
DEFAULT_RUN = "grpo"
KEEP_OPTIM_NAME = "optim.pt"
ITER_NAME = "ITER.txt"
SKIP_COPY_NAMES = ("model.safetensors", "train_config.json")
N_RENDERERS = 8
# ======Settings=========


class _ScenePrefetch:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._jobs: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            seed = self._jobs.get()
            if seed is None:
                return
            dest = self.root / f"{int(seed):08d}"
            dest.mkdir(parents=True, exist_ok=True)
            try:
                meta = generate(
                    seed_override=int(seed),
                    output_dir_override=dest,
                    write_previews=False,
                )
                self._ready.put((dest, meta))
            except BaseException as exc:
                self._ready.put(exc)

    def request(self, seed: int) -> None:
        self._jobs.put(int(seed))

    def take(self) -> tuple[Path, dict]:
        payload = self._ready.get()
        if isinstance(payload, BaseException):
            raise payload
        return payload

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5.0)


def _row2d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dim() >= 2 else tensor.unsqueeze(0)


def _collate_chunks(
    obs_list: list[dict],
    sde_list: list[SdeRollout],
    advantages: torch.Tensor,
    device: torch.device,
) -> tuple[dict, SdeRollout, torch.Tensor]:
    obs = {
        "images": torch.stack([item["images"] for item in obs_list], dim=0),
        "n_real": int(obs_list[0]["n_real"]),
        "tokens": torch.cat([_row2d(item["tokens"]) for item in obs_list], dim=0),
        "mask": torch.cat([_row2d(item["mask"]) for item in obs_list], dim=0),
        "state": torch.cat([item["state"] for item in obs_list], dim=0),
    }
    logp_old = torch.cat(
        [
            item.step_logprobs
            if item.step_logprobs.ndim == 2
            else item.step_logprobs.view(item.step_logprobs.shape[0], 1)
            for item in sde_list
        ],
        dim=1,
    )
    packed = SdeRollout(
        action=torch.cat([item.action for item in sde_list], dim=0).to(device),
        traj=torch.cat([item.traj for item in sde_list], dim=1).to(device),
        taus=sde_list[0].taus.to(device),
        dtau=sde_list[0].dtau,
        step_logprobs=logp_old.to(device),
        step_mask=torch.stack([item.step_mask for item in sde_list], dim=1).to(device),
        step_weights=sde_list[0].step_weights.to(device),
    )
    return obs, packed, advantages.to(device)


@dataclass
class GrpoHooks:
    should_stop: callable
    log: callable
    preview: callable | None = None
    metrics: callable | None = None
    group: callable | None = None
    progress: callable | None = None
    scene: callable | None = None


def _export_lerobot_weights(policy) -> dict[str, torch.Tensor]:
    exported = {}
    for name, tensor in policy.state_dict().items():
        if ".qkv_proj." in name:
            continue
        exported[f"model.{name}"] = tensor.detach().cpu().contiguous()
    for module_name, module in policy.named_modules():
        attn = getattr(module, "self_attn", None)
        if attn is None or not hasattr(attn, "qkv_proj"):
            continue
        fused = attn.qkv_proj
        q_out = int(attn.q_out)
        k_out = int(attn.k_out)
        base = f"model.{module_name}.self_attn" if module_name else "model.self_attn"
        weights = fused.weight.detach().cpu().split((q_out, k_out, k_out), dim=0)
        exported[f"{base}.q_proj.weight"] = weights[0].contiguous()
        exported[f"{base}.k_proj.weight"] = weights[1].contiguous()
        exported[f"{base}.v_proj.weight"] = weights[2].contiguous()
        if fused.bias is not None:
            biases = fused.bias.detach().cpu().split((q_out, k_out, k_out), dim=0)
            exported[f"{base}.q_proj.bias"] = biases[0].contiguous()
            exported[f"{base}.k_proj.bias"] = biases[1].contiguous()
            exported[f"{base}.v_proj.bias"] = biases[2].contiguous()
    return exported


def _write_camera_meta(model_dir: Path, cameras: list[str]) -> None:
    rename = policy_rename_map(cameras)
    for fname in ("train_config.json", "policy_preprocessor.json"):
        path = model_dir / fname
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if fname == "train_config.json":
            data["rename_map"] = rename
        else:
            for step in data.get("steps") or []:
                cfg = (step or {}).get("config") or {}
                if cfg.get("rename_map") is not None:
                    cfg["rename_map"] = rename
        path.write_text(json.dumps(data, indent=2) + "\n")


def _write_grpo_train_config(model_dir: Path, update: int) -> None:
    path = model_dir / "train_config.json"
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}
    payload["kind"] = "grpo"
    payload["update"] = int(update)
    payload.pop("steps", None)
    payload.pop("save_freq", None)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _save_checkpoint(actor, source: str, dest: Path, update: int) -> None:
    from safetensors.torch import save_file

    model_dir = dest / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(_export_lerobot_weights(actor.policy), str(model_dir / "model.safetensors"))
    from core.config import resolve_source

    resolved = resolve_source(source)
    src = resolved if isinstance(resolved, Path) else Path(source)
    if src.is_dir():
        for path in src.glob("*"):
            if path.name in SKIP_COPY_NAMES or not path.is_file():
                continue
            target = model_dir / path.name
            if not target.is_file():
                shutil.copy2(path, target)
    elif actor.engine.action_mean is not None:
        save_file(
            {
                "so100.buffer.action.mean": actor.engine.action_mean.detach().cpu(),
                "so100.buffer.action.std": actor.engine.action_std.detach().cpu(),
                "so100.buffer.observation.state.mean": actor.engine.state_mean.detach().cpu(),
                "so100.buffer.observation.state.std": actor.engine.state_std.detach().cpu(),
            },
            str(model_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        )
    _write_grpo_train_config(model_dir, update)
    cameras = list(getattr(actor.engine, "cameras", []) or [])
    if cameras:
        _write_camera_meta(model_dir, cameras)


def _trim_checkpoints(root: Path, keep_last: int) -> None:
    numbered = sorted(
        item
        for item in root.iterdir()
        if item.is_dir() and item.name.isdigit()
    )
    extra = numbered[: max(0, len(numbered) - max(1, keep_last))]
    for item in extra:
        shutil.rmtree(item, ignore_errors=True)


def _param_groups(policy, freeze_vlm: bool, expert_lr: float, vlm_lr: float):
    if freeze_vlm:
        params = [item for item in policy.parameters() if item.requires_grad]
        if not params:
            raise RuntimeError("no trainable parameters")
        return [{"params": params, "lr": expert_lr}]
    vlm = []
    expert = []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if "vlm_with_expert.vlm" in name:
            vlm.append(param)
        else:
            expert.append(param)
    groups = []
    if expert:
        groups.append({"params": expert, "lr": expert_lr})
    if vlm:
        groups.append({"params": vlm, "lr": vlm_lr})
    if not groups:
        raise RuntimeError("no trainable parameters")
    return groups


def _flatten(groups: list[list[EpisodeRecord]]):
    obs = []
    sdes = []
    advantages = []
    for group in groups:
        rewards = torch.tensor(
            [float(item.scored["total_reward"]) for item in group],
            dtype=torch.float32,
        )
        adv = group_advantages(rewards)
        for episode, value in zip(group, adv, strict=True):
            episode.advantage = float(value)
            scale = 1.0 / max(episode.n_chunks, 1) if EPISODE_EQUAL_WEIGHT else 1.0
            for chunk in episode.chunks:
                obs.append(chunk.obs)
                sdes.append(chunk.sde)
                advantages.append(float(value) * scale)
    return obs, sdes, torch.tensor(advantages, dtype=torch.float32)


def run_grpo(
    config: dict,
    checkpoint: str,
    output_dir: Path,
    device: torch.device,
    hooks: GrpoHooks,
    scene_dir: Path,
    llm_log_path: Path,
) -> None:
    from core.environment import RandomSceneEnv, open_scene_envs, retarget_scene_envs

    grpo = config["environment"]["grpo"]
    flow = grpo["flow"]
    opt = grpo["optimization"]
    ckpt = grpo["checkpointing"]
    smoke = grpo.get("smoke") or {}
    if smoke.get("enabled"):
        grpo = dict(grpo)
        grpo["scenes_per_update"] = smoke.get("scenes_per_update", grpo["scenes_per_update"])
        grpo["group_size"] = smoke.get("group_size", grpo["group_size"])
        config = dict(config)
        config["environment"] = dict(config["environment"])
        config["environment"]["rollout"] = dict(config["environment"]["rollout"])
        config["environment"]["rollout"]["duration_seconds"] = smoke.get(
            "duration_seconds",
            config["environment"]["rollout"]["duration_seconds"],
        )
        config["environment"]["grpo"] = grpo
    freeze_vlm = str(grpo.get("train_scope") or "experts") != "full_vla"
    group_size = max(2, int(grpo["group_size"]))
    parallel = max(1, min(int(grpo.get("parallel_rollouts") or 1), group_size))
    scenes_per_update = max(1, int(grpo["scenes_per_update"]))
    raw_waves = grpo.get("scene_waves")
    if raw_waves in (None, ""):
        scene_waves = scenes_per_update
    else:
        scene_waves = max(1, min(int(raw_waves), scenes_per_update))
    if scene_waves > 1:
        pool_n = min(group_size, parallel)
    else:
        pool_n = min(group_size, parallel * 2 if parallel > 1 else parallel)
    max_updates = grpo.get("max_updates")
    max_updates = None if max_updates in (None, "", 0) else max(1, int(max_updates))
    n_action_steps = max(1, int(grpo.get("n_action_steps") or 50))
    clip_eps = float(grpo.get("clip_eps", 0.2))
    kl_coef = float(grpo.get("kl_coef", DEFAULT_KL_COEF))
    micro = int(
        opt["update_microbatch_expert_only"]
        if freeze_vlm
        else opt["update_microbatch_full_vla"]
    )
    micro = max(1, micro)

    from model.act import is_act_checkpoint
    from core.config import normalize_policy_mode

    mode = normalize_policy_mode(config.get("policy_mode"))
    if mode == "act":
        raise RuntimeError("GRPO is not available in ACT mode.")
    cameras = policy_cameras_from_config(config)
    hooks.log(f"Load custom SmolVLA loop {checkpoint} on {device}.")
    engine, resolved = load_engine(
        checkpoint, device, on_log=hooks.log, cameras=cameras or None
    )
    if is_act_checkpoint(resolved):
        raise RuntimeError("GRPO does not support ACT checkpoints.")
    actor = FlowSdePolicy(
        engine,
        num_steps=int(flow["denoising_steps"]),
        noise_level=float(flow["noise_level"]),
        freeze_vlm=freeze_vlm,
        sde_mode=str(flow.get("sde_mode") or "one_random_step"),
        flow_scale=dict(flow.get("flow_scale") or {}),
    )
    hooks.log("Load frozen SFT reference for KL and drift.")
    ref_engine, _ = load_engine(
        resolved, device, on_log=hooks.log, cameras=cameras or None
    )
    reference = FlowSdePolicy(
        ref_engine,
        num_steps=actor.num_steps,
        noise_level=actor.noise_level,
        freeze_vlm=False,
        sde_mode=actor.sde_mode,
        flow_scale=actor.flow_scale,
    )
    for param in reference.parameters():
        param.requires_grad_(False)
    reference.eval()
    trainable = [item for item in actor.parameters() if item.requires_grad]
    optim = torch.optim.AdamW(
        _param_groups(
            actor.policy,
            freeze_vlm,
            float(opt["action_expert_learning_rate"]),
            float(opt["vlm_learning_rate"]),
        ),
        weight_decay=float(opt.get("weight_decay") or 0.0),
        betas=(float(opt["adam_beta1"]), float(opt["adam_beta2"])),
        eps=float(opt["adam_eps"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = output_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    scope = "experts only" if freeze_vlm else "full VLA"
    hooks.log(
        f"Flow-SDE GRPO · {actor.sde_mode} · {actor.num_steps} steps · "
        f"noise {actor.noise_level:g} · group {group_size} · "
        f"infer {parallel} · pool {pool_n} · micro {micro} · "
        f"{float(config['environment']['rollout']['duration_seconds']):g}s · "
        f"{scenes_per_update} scenes/update · {scene_waves} scene waves · {scope} · "
        f"KL {kl_coef:g}."
    )
    if cameras:
        hooks.log(
            "Camera map · "
            + ", ".join(
                f"camera{index}={name}" for index, name in enumerate(cameras, start=1)
            )
        )

    update = 0
    scenes_done = 0
    seed0 = int(config["seed"])
    vis = grpo.get("visualization") or {}
    preview_on = bool(vis.get("enabled", True)) and hooks.preview is not None

    def _preview(frames):
        names = list(frames)
        hooks.preview(names, [encode_jpeg(frames[name]) for name in names])

    scene_dir.mkdir(parents=True, exist_ok=True)
    prefetch = _ScenePrefetch(scene_dir)
    next_seed = seed0
    for _ in range(scene_waves):
        prefetch.request(next_seed)
        next_seed += 1
    pools: list[tuple] = []

    def _bind_pool(slot: int, scene_path: Path, metadata: dict):
        seeds = [int(metadata["seed"]) + index for index in range(pool_n)]
        if slot >= len(pools):
            if pool_n <= 1:
                pools.append(
                    ([], [RandomSceneEnv(scene_path, config, sensor_seed=seeds[0])])
                )
            else:
                _, renderers, envs = open_scene_envs(
                    scene_path,
                    config,
                    sensor_seeds=seeds,
                    n_renderers=min(N_RENDERERS, pool_n),
                )
                pools.append((renderers, envs))
        else:
            retarget_scene_envs(pools[slot][1], scene_path, sensor_seeds=seeds)
        return pools[slot][1]

    try:
      while not hooks.should_stop():
        if max_updates is not None and update >= max_updates:
            hooks.log("Reached max updates.")
            break
        actor.eval()
        groups: list[list[EpisodeRecord]] = []
        taken = 0
        while taken < scenes_per_update:
            if hooks.should_stop():
                break
            wave_n = min(scene_waves, scenes_per_update - taken)
            if hooks.progress is not None:
                hooks.progress(
                    "rollout",
                    scenes_done,
                    None if max_updates is None else max_updates * scenes_per_update,
                    f"Scenes {scenes_done + 1}–{scenes_done + wave_n} · generate",
                )
            items = []
            for _ in range(wave_n):
                hooks.log(f"Generate scene seed {seed0 + scenes_done + len(items)}.")
                items.append(prefetch.take())
                prefetch.request(next_seed)
                next_seed += 1
            if hooks.scene is not None:
                metadata = items[0][1]
                hooks.scene(
                    {
                        "seed": metadata.get("seed"),
                        "instruction": metadata.get("instruction"),
                        "cameras": list(metadata.get("cameras") or []),
                        "target": metadata.get("target"),
                    }
                )
            packs = []
            for slot, (scene_path, metadata) in enumerate(items):
                packs.append(
                    (
                        _bind_pool(slot, scene_path, metadata),
                        str(metadata.get("instruction") or ""),
                    )
                )
            t_roll = time.perf_counter()
            wave_groups = collect_scene_wave(
                packs,
                actor,
                config["reward"],
                n_action_steps,
                group_size,
                llm_log_path,
                hooks.should_stop,
                on_preview=_preview if preview_on else None,
                on_log=hooks.log,
                wave_size=parallel,
                preview_hz=float(vis.get("stream_fps") or 15),
            )
            roll_s = time.perf_counter() - t_roll
            if not any(wave_groups):
                break
            for group in wave_groups:
                if not group:
                    continue
                groups.append(group)
                scenes_done += 1
                taken += 1
                mean_r = sum(float(item.scored["total_reward"]) for item in group) / len(group)
                success = sum(1 for item in group if item.success) / len(group)
                hooks.log(
                    f"Scene {scenes_done} · reward {mean_r:.3f} · success {success:.2f} · "
                    f"{group[0].n_env_steps} steps (last) · wave {roll_s:.1f}s."
                )

        if hooks.should_stop() or not groups:
            break
        flat_groups = [_flatten([group]) for group in groups]
        keep = [i for i, group in enumerate(groups) if any(item.success for item in group)]
        if not keep:
            hooks.log("No successful groups, skip update.", "warn")
            continue
        if len(keep) < len(groups):
            hooks.log(f"Drop {len(groups) - len(keep)}/{len(groups)} groups with 0 success.")
            flat_groups = [flat_groups[i] for i in keep]
        if hooks.group is not None:
            hooks.group(
                [
                    {
                        "member": item.member,
                        "success": item.success,
                        "steps": item.n_env_steps,
                        "advantage": item.advantage,
                        **item.scored,
                    }
                    for group in groups
                    for item in group
                ]
            )
        obs_all: list[dict] = []
        sde_all: list[SdeRollout] = []
        adv_parts: list[torch.Tensor] = []
        for obs_list, sde_list, advantages in flat_groups:
            obs_all.extend(obs_list)
            sde_all.extend(sde_list)
            adv_parts.append(advantages)
        adv_all = torch.cat(adv_parts)
        n_chunks = len(obs_all)
        if n_chunks == 0:
            hooks.log("No chunks in this update, skip.", "warn")
            continue

        if hooks.progress is not None:
            hooks.progress("update", update, max_updates, f"GRPO update {update + 1}")
        actor.train()
        kl_sum = clip_sum = drift_sum = mag_sum = pg_sum = 0.0
        n_mb = 0
        grad_norm = 0.0
        ref_cache: list[torch.Tensor | None] = [None] * (
            (n_chunks + micro - 1) // micro
        )
        optim.zero_grad()
        for mb, start in enumerate(range(0, n_chunks, micro)):
            if hooks.should_stop():
                break
            stop = min(start + micro, n_chunks)
            obs, packed, adv = _collate_chunks(
                obs_all[start:stop],
                sde_all[start:stop],
                adv_all[start:stop],
                device,
            )
            logp_new = actor.recompute_logprobs(obs, packed)
            logp_old = packed.step_logprobs
            if ref_cache[mb] is None:
                with torch.no_grad():
                    ref_cache[mb] = reference.recompute_logprobs(obs, packed).detach()
            logp_ref = ref_cache[mb]
            keep = packed.step_mask
            if not keep.any():
                continue
            n_keep = int(keep.sum())
            adv_full = adv.view(1, -1).expand_as(logp_new)
            weight_full = packed.step_weights.view(-1, 1).expand_as(logp_new)
            out = grpo_loss(
                logp_new=logp_new[keep],
                logp_old=logp_old[keep],
                advantages=adv_full[keep],
                logp_ref=logp_ref[keep],
                step_weights=weight_full[keep],
                clip_eps=clip_eps,
                kl_coef=kl_coef,
            )
            (out.loss * (n_keep / n_chunks)).backward()
            pg_sum += float(out.policy_loss)
            kl_sum += float(out.kl_loss)
            clip_sum += float(out.clip_fraction)
            drift_sum += float(out.drift)
            mag_sum += float(out.loss_mag)
            n_mb += 1
        if hooks.should_stop():
            break
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, float(opt["max_grad_norm"]))
        )
        optim.step()
        optim.zero_grad()
        update += 1
        rewards = [float(item.scored["total_reward"]) for group in groups for item in group]
        successes = [1.0 if item.success else 0.0 for group in groups for item in group]
        info = {
            "update": update,
            "scenes": scenes_done,
            "success_rate": sum(successes) / max(len(successes), 1),
            "mean_reward": sum(rewards) / max(len(rewards), 1),
            "n_episodes": len(rewards),
            "n_chunks": n_chunks,
            "loss": mag_sum / max(n_mb, 1),
            "policy_loss": pg_sum / max(n_mb, 1),
            "kl": kl_sum / max(n_mb, 1),
            "clip_fraction": clip_sum / max(n_mb, 1),
            "drift": drift_sum / max(n_mb, 1),
            "grad_norm": grad_norm,
        }
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(info) + "\n")
        if hooks.metrics is not None:
            hooks.metrics(info)
        hooks.log(
            f"Update {update} · reward {info['mean_reward']:.3f} · "
            f"success {info['success_rate']:.2f} · loss {info['loss']:.4f} · "
            f"kl {info['kl']:.4f} · clip {info['clip_fraction']:.2f} · "
            f"drift {info['drift']:.3f}."
        )
        save_every = max(1, int(ckpt.get("save_every_scenes") or 30))
        if scenes_done % save_every == 0 or (max_updates is not None and update >= max_updates):
            step_dir = ckpt_root / f"{update:06d}"
            _save_checkpoint(actor, resolved, step_dir, update)
            (step_dir / "training_state").mkdir(exist_ok=True)
            (step_dir / "training_state" / "training_step.json").write_text(
                json.dumps({"step": update, "update": update, "kind": "grpo"})
            )
            last = ckpt_root / "last"
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(step_dir.name, target_is_directory=True)
            if ckpt.get("keep_optimizer"):
                torch.save(optim.state_dict(), step_dir / KEEP_OPTIM_NAME)
            (step_dir / ITER_NAME).write_text(f"{update}\n")
            _trim_checkpoints(ckpt_root, int(ckpt.get("keep_last") or 3))
            hooks.log(f"Saved {step_dir}.")
    finally:
        prefetch.close()
        for renderers, envs in pools:
            for env in envs:
                env.close()
            for renderer in renderers or []:
                renderer.close()
    hooks.log("GRPO stopped.")
