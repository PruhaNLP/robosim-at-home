"""Standalone ACT: ResNet encoder, transformer decoder, no language."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

_SIM_ROOT = Path(__file__).resolve().parents[1] / "sim"
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

from core.config import resolve_source

# ======Settings=========
IMAGE_SIZE = 256
CHUNK_SIZE = 50
ACTION_DIM = 6
NUM_CAMERAS = 2
CAMERA_ORDER = ("front", "wrist")
DIM = 768
N_HEADS = 12
N_ENC = 4
N_DEC = 1
FF = 3072
DROPOUT = 0.1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORM_EPS = 1e-6
TRAJ_NAME = "traj.npz"
CONFIG_NAME = "config.json"
STATS_NAME = "stats.safetensors"
WEIGHTS_NAME = "model.safetensors"
# ======Settings=========


def _unwrap_pretrained(path: Path) -> Path:
    nested = path / "pretrained_model"
    if (nested / WEIGHTS_NAME).is_file() or (nested / CONFIG_NAME).is_file():
        return nested
    return path


def _existing_dir(source: str) -> Path | None:
    raw = str(source or "").strip()
    if not raw:
        return None
    try:
        resolved = resolve_source(raw)
        if isinstance(resolved, Path) and resolved.is_dir():
            return resolved
    except ValueError:
        pass
    path = Path(raw)
    return path if path.is_dir() else None


def is_act_scratch(source: str) -> bool:
    return str(source or "").strip().lower() in {"", "act", "lerobot/act"}


def is_act_checkpoint(source: str) -> bool:
    if is_act_scratch(source):
        return True
    path = _existing_dir(source)
    if path is None:
        return False
    cfg = _unwrap_pretrained(path) / CONFIG_NAME
    if cfg.is_file():
        data = json.loads(cfg.read_text())
        return str(data.get("type") or "") == "act"
    return False


def _resize_pad(image: Tensor, size: int) -> Tensor:
    _, _, current_h, current_w = image.shape
    if current_h == size and current_w == size:
        return image
    ratio = max(current_w / size, current_h / size)
    new_h = max(1, int(current_h / ratio))
    new_w = max(1, int(current_w / ratio))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = size - new_h
    pad_w = size - new_w
    return F.pad(resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))


def _configure_backends(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class ActPolicy(nn.Module):
    def __init__(self, n_cameras: int = NUM_CAMERAS) -> None:
        super().__init__()
        self.n_cameras = max(1, int(n_cameras))
        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.image_proj = nn.Conv2d(512, DIM, kernel_size=1)
        self.state_proj = nn.Linear(ACTION_DIM, DIM)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=DIM,
                nhead=N_HEADS,
                dim_feedforward=FF,
                dropout=DROPOUT,
                batch_first=True,
                activation="relu",
            ),
            num_layers=N_ENC,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=DIM,
                nhead=N_HEADS,
                dim_feedforward=FF,
                dropout=DROPOUT,
                batch_first=True,
                activation="relu",
            ),
            num_layers=N_DEC,
        )
        self.query = nn.Embedding(CHUNK_SIZE, DIM)
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.n_cameras * 64, DIM))
        self.action_head = nn.Linear(DIM, ACTION_DIM)
        nn.init.normal_(self.pos, std=0.02)

    def _images(self, images: Tensor) -> Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        batch, cams, _, height, width = images.shape
        flat = images.reshape(batch * cams, 3, height, width)
        flat = _resize_pad(flat, IMAGE_SIZE)
        mean = flat.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = flat.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        flat = (flat - mean) / std
        feats = self.image_proj(self.backbone(flat)["feature_map"])
        _, dim, grid_h, grid_w = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)
        return tokens.reshape(batch, cams * grid_h * grid_w, dim)

    def forward(self, images: Tensor, state: Tensor) -> Tensor:
        vision = self._images(images)
        robot = self.state_proj(state).unsqueeze(1)
        memory = torch.cat([robot, vision], dim=1)
        memory = memory + self.pos[:, : memory.size(1)]
        memory = self.encoder(memory)
        query = self.query.weight.unsqueeze(0).expand(state.size(0), -1, -1)
        hidden = self.decoder(query, memory)
        return self.action_head(hidden)


class ActEngine:
    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy: ActPolicy | None = None
        self.cameras: list[str] = list(CAMERA_ORDER[:NUM_CAMERAS])
        self.state_mean: Tensor | None = None
        self.state_std: Tensor | None = None
        self.action_mean: Tensor | None = None
        self.action_std: Tensor | None = None
        self.source = "act"

    def _set_stats(
        self,
        state_mean: Tensor,
        state_std: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
    ) -> None:
        self.state_mean = state_mean.to(self.device).float().view(-1)[:ACTION_DIM]
        self.state_std = state_std.to(self.device).float().view(-1)[:ACTION_DIM].clamp_min(NORM_EPS)
        self.action_mean = action_mean.to(self.device).float().view(-1)[:ACTION_DIM]
        self.action_std = action_std.to(self.device).float().view(-1)[:ACTION_DIM].clamp_min(NORM_EPS)

    def create(
        self,
        cameras: list[str],
        state_mean: Tensor,
        state_std: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
    ) -> None:
        _configure_backends(self.device)
        self.cameras = [name for name in CAMERA_ORDER if name in cameras] or list(cameras)
        self.policy = ActPolicy(n_cameras=len(self.cameras)).to(self.device)
        self._set_stats(state_mean, state_std, action_mean, action_std)
        self.source = "act"

    def load(self, source: str, on_log=None) -> None:
        def log(text: str) -> None:
            if on_log is not None:
                on_log(text)

        path = _existing_dir(source)
        if path is None:
            raw = str(source or "").strip()
            if is_act_scratch(raw):
                raise RuntimeError(f"ACT checkpoint must be a folder, got {source}")
            log(f"Download ACT {raw} from Hugging Face.")
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(repo_id=raw))
            if not path.is_dir():
                raise RuntimeError(f"ACT checkpoint must be a folder, got {source}")
        path = _unwrap_pretrained(path)
        cfg_path = path / CONFIG_NAME
        if not cfg_path.is_file():
            raise RuntimeError(f"no {CONFIG_NAME} in {path}")
        config = json.loads(cfg_path.read_text())
        if str(config.get("type") or "") != "act":
            raise RuntimeError(f"{path} is not an ACT checkpoint")
        weights = path / WEIGHTS_NAME
        stats_path = path / STATS_NAME
        if not weights.is_file() or not stats_path.is_file():
            raise RuntimeError(f"ACT checkpoint is missing weights or stats in {path}")
        _configure_backends(self.device)
        self.cameras = [str(name) for name in (config.get("cameras") or CAMERA_ORDER[:NUM_CAMERAS])]
        log(f"Loading ACT {path} on {self.device}…")
        self.policy = ActPolicy(n_cameras=len(self.cameras)).to(self.device)
        self.policy.load_state_dict(load_file(str(weights)))
        stats = load_file(str(stats_path))
        self._set_stats(
            stats["observation.state.mean"],
            stats["observation.state.std"],
            stats["action.mean"],
            stats["action.std"],
        )
        self.source = str(path)
        log("ACT ready")

    def save(self, dest: Path) -> None:
        if self.policy is None or self.action_mean is None:
            raise RuntimeError("ACT engine is empty")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / CONFIG_NAME).write_text(
            json.dumps(
                {
                    "type": "act",
                    "chunk_size": CHUNK_SIZE,
                    "image_size": IMAGE_SIZE,
                    "dim": DIM,
                    "cameras": list(self.cameras),
                },
                indent=2,
            )
            + "\n"
        )
        save_file({key: value.detach().cpu() for key, value in self.policy.state_dict().items()}, str(dest / WEIGHTS_NAME))
        save_file(
            {
                "observation.state.mean": self.state_mean.detach().cpu(),
                "observation.state.std": self.state_std.detach().cpu(),
                "action.mean": self.action_mean.detach().cpu(),
                "action.std": self.action_std.detach().cpu(),
            },
            str(dest / STATS_NAME),
        )

    def _stack_frames(self, frames: dict[str, Image.Image]) -> Tensor:
        cpu = []
        for name in self.cameras:
            if name in frames:
                cpu.append(torch.from_numpy(np.asarray(frames[name].convert("RGB"))))
            else:
                cpu.append(torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 3, dtype=torch.uint8))
        stacked = torch.stack(cpu, dim=0).to(device=self.device, non_blocking=True)
        stacked = stacked.permute(0, 3, 1, 2).to(dtype=torch.float32).mul_(1.0 / 255.0)
        return stacked

    def prepare_obs(
        self,
        frames: dict[str, Image.Image],
        state: np.ndarray,
        instruction: str = "",
    ) -> dict:
        if self.policy is None or self.state_mean is None:
            raise RuntimeError("ACT engine is not loaded")
        images = self._stack_frames(frames)
        joints = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(-1)[:ACTION_DIM]
        state_tensor = ((joints - self.state_mean) / self.state_std).unsqueeze(0)
        return {
            "images": images.unsqueeze(0),
            "n_real": int(images.size(0)),
            "state": state_tensor,
        }

    def normalize_actions(self, actions: Tensor) -> Tensor:
        return (actions - self.action_mean) / self.action_std

    def actions_to_env(self, actions: Tensor) -> np.ndarray:
        chunk = actions[0, :, :ACTION_DIM] * self.action_std + self.action_mean
        chunk = torch.nan_to_num(chunk.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return chunk.detach().cpu().numpy()

    def predict_mean(self, obs: dict) -> Tensor:
        if self.policy is None:
            raise RuntimeError("ACT engine is not loaded")
        return self.policy(obs["images"], obs["state"])

    @torch.inference_mode()
    def predict_chunk(
        self,
        frames: dict[str, Image.Image],
        state: np.ndarray,
        instruction: str = "",
    ) -> np.ndarray:
        obs = self.prepare_obs(frames, state, instruction)
        return self.actions_to_env(self.predict_mean(obs))


def _tensor_stats(states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        states.mean(0),
        states.std(0).clamp_min(NORM_EPS),
        actions.mean(0),
        actions.std(0).clamp_min(NORM_EPS),
    )


def _decode_mp4(path: Path) -> Tensor:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        ]
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"empty video {path}")
    return torch.stack(frames, 0)


def _resize_cache(frames: Tensor, device: torch.device) -> Tensor:
    out = []
    for start in range(0, frames.size(0), 64):
        chunk = frames[start : start + 64].permute(0, 3, 1, 2).to(
            device=device, dtype=torch.float32
        ).div_(255.0)
        chunk = _resize_pad(chunk, IMAGE_SIZE)
        out.append((chunk * 255.0).clamp_(0, 255).to(torch.uint8).cpu())
    return torch.cat(out, 0)


def _load_collect_cache(
    episodes: list[dict],
    cameras: list[str],
    device: torch.device,
    log,
    progress=None,
    should_stop=None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    images = []
    states = []
    actions = []
    ends = []
    offset = 0
    total = len(episodes)
    for index, item in enumerate(episodes, start=1):
        if should_stop is not None and should_stop():
            break
        label = f"{item.get('repoId', 'local')} #{item.get('index', index)}"
        if progress is not None:
            progress("prepare", index, total, f"Decode {label}")
        traj = np.load(item["folder"] / TRAJ_NAME)
        state = torch.as_tensor(np.asarray(traj["state"], dtype=np.float32)[:, :ACTION_DIM])
        action = torch.as_tensor(np.asarray(traj["action"], dtype=np.float32)[:, :ACTION_DIM])
        length = int(action.size(0))
        cams = []
        for name in cameras:
            path = Path(item["folder"]) / f"{name}.mp4"
            if not path.is_file():
                raise RuntimeError(f"{label} has no {name}.mp4")
            log(f"Decode {label} {name}…")
            frames = _resize_cache(_decode_mp4(path), device)
            if frames.size(0) != length:
                raise RuntimeError(
                    f"{label} {name}: {frames.size(0)} frames, trajectory {length}"
                )
            cams.append(frames)
        images.append(torch.stack(cams, dim=1))
        states.append(state)
        actions.append(action)
        offset += length
        ends.append(torch.full((length,), offset, dtype=torch.int64))
    if not states:
        raise RuntimeError("no ACT frames loaded")
    return (
        torch.cat(images, 0),
        torch.cat(states, 0),
        torch.cat(actions, 0),
        torch.cat(ends, 0),
    )


def _act_lr(
    step: int,
    peak: float,
    final: float,
    warmup: int,
    decay: int,
    total: int,
) -> float:
    step = max(1, int(step))
    warmup = max(0, int(warmup))
    decay = max(0, int(decay))
    total = max(1, int(total))
    if warmup + decay > total:
        if warmup >= total:
            return peak * step / total
        decay = total - warmup
    if warmup > 0 and step <= warmup:
        return peak * step / warmup
    start = total - decay
    if decay > 0 and step > start:
        progress = min(1.0, (step - start) / decay)
        return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * progress))
    return peak


class ActRamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: Tensor,
        states: Tensor,
        actions: Tensor,
        ends: Tensor,
    ) -> None:
        self.images = images
        self.states = states
        self.actions = actions
        self.ends = ends

    def __len__(self) -> int:
        return int(self.states.size(0))

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        take = min(CHUNK_SIZE, end - index)
        chunk = torch.zeros(CHUNK_SIZE, ACTION_DIM, dtype=torch.float32)
        mask = torch.zeros(CHUNK_SIZE, dtype=torch.float32)
        chunk[:take] = self.actions[index : index + take]
        mask[:take] = 1.0
        return self.images[index].float().div_(255.0), self.states[index], chunk, mask


def run_act_sft(
    episodes: list[dict],
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch: int,
    lr: float,
    save_every: int,
    source: str,
    cameras: list[str],
    log,
    progress,
    should_stop,
    on_metrics=None,
    training: dict | None = None,
) -> None:
    training = training or {}
    cameras = [name for name in CAMERA_ORDER if name in cameras] or list(CAMERA_ORDER[:1])
    images, states, actions, ends = _load_collect_cache(
        episodes,
        cameras,
        device,
        log,
        progress=progress,
        should_stop=should_stop,
    )
    if should_stop():
        return
    state_mean, state_std, action_mean, action_std = _tensor_stats(states, actions)
    engine = ActEngine(device=str(device))
    if is_act_scratch(source):
        log("Train ACT from scratch.")
        engine.create(cameras, state_mean, state_std, action_mean, action_std)
    else:
        engine.load(source, on_log=log)
        cameras = list(engine.cameras)
    if images.size(1) != len(engine.cameras):
        raise RuntimeError(
            f"cached {images.size(1)} cameras, ACT engine has {len(engine.cameras)}"
        )
    log(
        f"Frame cache {images.size(0)} frames · {images.size(1)} cameras · "
        f"{IMAGE_SIZE}px · {images.nbytes / 1024**3:.2f} GB."
    )
    loader = torch.utils.data.DataLoader(
        ActRamDataset(images, states, actions, ends),
        batch_size=max(1, int(batch)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    steps = max(1, epochs * max(1, math.ceil(len(loader.dataset) / max(1, int(batch)))))
    progress("train", 0, steps, f"Start ACT · {steps} steps")
    peak_lr = float(lr)
    decay_lr = max(0.0, float(training.get("scheduler_decay_lr", 2.5e-6)))
    warmup = max(0, int(training.get("scheduler_warmup_steps", 1000)))
    decay_steps = max(0, int(training.get("scheduler_decay_steps", 30000)))
    weight_decay = max(0.0, float(training.get("optimizer_weight_decay", 1e-4)))
    beta1 = float(training.get("optimizer_beta1", 0.9))
    beta2 = float(training.get("optimizer_beta2", 0.95))
    eps = max(0.0, float(training.get("optimizer_eps", 1e-8)))
    grad_clip = max(0.0, float(training.get("grad_clip_norm", 10.0)))
    use_amp = bool(training.get("use_amp", True)) and device.type == "cuda"
    amp_dtype = torch.float32
    if use_amp:
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    optim = torch.optim.AdamW(
        engine.policy.parameters(),
        lr=peak_lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps,
    )
    log(
        f"ACT optim · amp {amp_dtype if use_amp else 'off'} · "
        f"warmup {warmup} · flat · decay {decay_steps} → {decay_lr:g}."
    )
    step = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = output_dir / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def write_ckpt(name: str) -> Path:
        dest = ckpt_root / name / "pretrained_model"
        engine.save(dest)
        last = ckpt_root / "last"
        if last.exists() or last.is_symlink():
            last.unlink()
        last.symlink_to(name, target_is_directory=True)
        return dest

    engine.policy.train()
    try:
        for epoch in range(1, epochs + 1):
            if should_stop():
                break
            for images_b, state, action_b, mask in loader:
                if should_stop():
                    break
                step += 1
                lr_now = _act_lr(step, peak_lr, decay_lr, warmup, decay_steps, steps)
                for group in optim.param_groups:
                    group["lr"] = lr_now
                images_b = images_b.to(device, non_blocking=True)
                state = (state.to(device, non_blocking=True) - engine.state_mean) / engine.state_std
                target = engine.normalize_actions(action_b.to(device, non_blocking=True))
                mask = mask.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred = engine.policy(images_b, state)
                    per = (pred.float() - target).abs().mean(dim=-1)
                    loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optim)
                    grad = float(torch.nn.utils.clip_grad_norm_(engine.policy.parameters(), grad_clip))
                else:
                    grad = 0.0
                scaler.step(optim)
                scaler.update()
                if on_metrics is not None:
                    on_metrics(
                        {
                            "step": step,
                            "loss": float(loss),
                            "lr": lr_now,
                            "grdn": grad,
                            "epch": epoch,
                            "smp/s": step / max(time.monotonic() - started, 1e-6),
                        }
                    )
                if step == 1 or step % 10 == 0 or step >= steps:
                    log(
                        f"step:{step} smpl:{step * images_b.size(0)} epch:{epoch} "
                        f"loss:{float(loss):.4f} grdn:{grad:.3f} lr:{lr_now:g}"
                    )
                progress("train", step, steps, f"Step {step}/{steps}")
            if save_every > 0 and epoch % save_every == 0 and step > 0:
                dest = write_ckpt(f"{step:06d}")
                log(f"Saved {dest}.")
    finally:
        if step > 0:
            dest = write_ckpt(f"{step:06d}")
            log(f"Saved {dest}.")


def _normalize_camera_names(value) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def run_act_test(
    scene_dir: Path,
    config: dict,
    duration_seconds: float,
    n_action_steps: int,
    on_frames,
    should_stop,
    on_log=None,
    view_fps: float = 15,
    on_state=None,
    checkpoint: str | None = None,
    cameras=None,
) -> dict:
    from core.environment import RandomSceneEnv

    def log(text: str, kind: str = "info") -> None:
        if on_log is not None:
            on_log(text, kind)

    if not checkpoint:
        raise RuntimeError("ACT inference needs a trained ACT checkpoint")
    engine = ActEngine()
    engine.load(checkpoint, on_log=log)
    if not cameras:
        from core.config import policy_cameras_from_config

        cameras = policy_cameras_from_config(config)
    chosen = _normalize_camera_names(cameras)
    if chosen:
        trained = list(engine.cameras)
        n_slots = len(trained)
        engine.cameras = chosen[:n_slots]
        while len(engine.cameras) < n_slots:
            engine.cameras.append(f"__empty_{len(engine.cameras)}")
        if len(chosen) != n_slots:
            log(f"ACT checkpoint has {n_slots} camera slots.")
    rollout = dict(config)
    rollout["environment"] = dict(config["environment"])
    rollout["environment"]["rollout"] = dict(config["environment"]["rollout"])
    rollout["environment"]["rollout"]["duration_seconds"] = float(duration_seconds)
    env = RandomSceneEnv(scene_dir, rollout, sensor_seed=int(config["seed"]))
    try:
        log("Reset scene…")
        observation = env.reset()
        frames = {name: env.render_camera(name) for name in env.camera_names}
        on_frames(frames)
        steps = 0
        max_steps = max(1, int(float(duration_seconds) * env.control_hz))
        take = max(1, min(int(n_action_steps), CHUNK_SIZE))
        control_dt = 1.0 / env.control_hz
        view_dt = 1.0 / max(1.0, float(view_fps))
        last_view = 0.0
        log(
            f"ACT rollout {max_steps} steps at {env.control_hz:g} Hz, "
            f"chunk {take}, cameras {', '.join(engine.cameras)}."
        )

        def publish_state(force: bool = False) -> None:
            nonlocal last_view
            now = time.monotonic()
            if not force and last_view and now - last_view < view_dt:
                return
            last_view = now
            if on_state is not None:
                on_state(env.data.qpos.copy(), env.data.qvel.copy())
            on_frames({name: env.render_camera(name) for name in env.camera_names})

        publish_state(force=True)
        chunk_index = 0
        while steps < max_steps and not should_stop():
            chunk_index += 1
            frames = {name: env.render_camera(name) for name in env.camera_names}
            log(f"Infer #{chunk_index} at step {steps}/{max_steps}.")
            started = time.perf_counter()
            chunk = engine.predict_chunk(
                frames,
                np.asarray(observation["state"], dtype=np.float32),
            )
            log(f"Infer #{chunk_index} done {(time.perf_counter() - started) * 1000.0:.0f} ms.")
            next_tick = time.perf_counter()
            for action in chunk[:take]:
                if steps >= max_steps or should_stop():
                    break
                observation, done = env.step(action, render_images=False)
                steps += 1
                publish_state()
                next_tick += control_dt
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.perf_counter()
                if done:
                    publish_state(force=True)
                    reward = env.finalize_reward()
                    log(f"Success at step {steps}/{max_steps}.")
                    return {
                        "success": bool(reward.success),
                        "steps": steps,
                        "seconds": steps / env.control_hz,
                        "qpos": env.data.qpos.copy(),
                        "qvel": env.data.qvel.copy(),
                    }
        reward = env.finalize_reward()
        return {
            "success": bool(reward.success),
            "steps": steps,
            "seconds": steps / env.control_hz,
            "qpos": env.data.qpos.copy(),
            "qvel": env.data.qvel.copy(),
        }
    finally:
        env.close()
