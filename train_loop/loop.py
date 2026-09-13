from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    push_checkpoint_to_hub,
    save_checkpoint,
    should_save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.jobs import submit_to_hf
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.import_utils import _peft_available, require_package
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from train_loop.amp import make_scaler, resolve_amp
from train_loop.cameras import (
    PREFETCH_FRAMES,
    EpisodeStreamDataset,
    variable_camera_collate,
    wrap_dataset,
)

if _peft_available:
    from peft import PeftModel
else:
    PeftModel = None

from lerobot.scripts.lerobot_eval import eval_policy_all

# ======Settings=========
IMAGE_PREFIX = "observation.images."
VISION_CACHE_ENABLED = True
VISION_CACHE_GB = 16
VISION_CACHE_ENV = "ROBOSIM_VISION_CACHE"
VISION_CACHE_GB_ENV = "ROBOSIM_VISION_CACHE_GB"
# ======Settings=========


def _vision_cache_config() -> tuple[bool, int]:
    raw_enabled = os.environ.get(VISION_CACHE_ENV)
    if raw_enabled is None:
        enabled = VISION_CACHE_ENABLED
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "no", "off"}
    raw_gb = os.environ.get(VISION_CACHE_GB_ENV)
    gb = float(raw_gb) if raw_gb is not None else float(VISION_CACHE_GB)
    if gb <= 0:
        enabled = False
    return enabled, max(0, int(gb * 1024**3))


class FrozenVisionCache:
    """Reuse SigLIP embeddings while the vision encoder is frozen."""

    def __init__(self, embed_fn, bytes_limit: int) -> None:
        self.embed_fn = embed_fn
        self.cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.cache_bytes = 0
        self.bytes_limit = int(bytes_limit)
        self.ids = None
        self.cam_i = 0
        self.hits = 0
        self.misses = 0

    def set_batch(self, batch: dict) -> None:
        episode = batch.get("episode_index")
        frame = batch.get("frame_index")
        if episode is None or frame is None:
            self.ids = None
        else:
            self.ids = (
                episode.detach().cpu().tolist(),
                frame.detach().cpu().tolist(),
            )
        self.cam_i = 0

    def _store(self, key: tuple, item: torch.Tensor) -> None:
        nbytes = item.numel() * item.element_size()
        while self.cache and self.cache_bytes + nbytes > self.bytes_limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= old.numel() * old.element_size()
        self.cache[key] = item
        self.cache_bytes += nbytes

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.ids is None:
            return self.embed_fn(image)
        episode, frame = self.ids
        outputs: list[torch.Tensor | None] = [None] * image.shape[0]
        missing: list[int] = []
        for index in range(image.shape[0]):
            key = (int(episode[index]), int(frame[index]), self.cam_i)
            cached = self.cache.get(key)
            if cached is None:
                missing.append(index)
            else:
                self.cache.move_to_end(key)
                outputs[index] = cached.to(device=image.device, dtype=image.dtype)
                self.hits += 1
        if missing:
            computed = self.embed_fn(image[missing])
            for offset, index in enumerate(missing):
                key = (int(episode[index]), int(frame[index]), self.cam_i)
                item = computed[offset].detach().to("cpu")
                self._store(key, item)
                outputs[index] = computed[offset]
                self.misses += 1
        self.cam_i += 1
        return torch.stack(outputs, dim=0)


def _install_vision_cache(policy, bytes_limit: int) -> FrozenVisionCache | None:
    model = getattr(policy, "model", None)
    vlm = getattr(model, "vlm_with_expert", None) if model is not None else None
    if vlm is None or not hasattr(vlm, "embed_image"):
        return None
    cache = FrozenVisionCache(vlm.embed_image, bytes_limit)
    vlm.embed_image = cache
    return cache


@contextmanager
def _make_eval_envs(cfg: TrainPipelineConfig) -> Iterator[dict[str, dict[int, Any]]]:
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
    )
    try:
        yield envs
    finally:
        close_envs(envs)


def _dataloader_worker_kwargs(cfg: TrainPipelineConfig) -> dict[str, Any]:
    workers_enabled = cfg.num_workers > 0
    return {
        "prefetch_factor": cfg.prefetch_factor if workers_enabled else None,
        "persistent_workers": cfg.persistent_workers and workers_enabled,
        "multiprocessing_context": cfg.dataloader_multiprocessing_context
        if workers_enabled
        else None,
    }


def _worker_init(_worker_id: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _move_images(batch: dict, device: torch.device, camera_keys: list[str]) -> None:
    for key in camera_keys:
        tensor = batch.get(key)
        if tensor is None or not torch.is_tensor(tensor):
            continue
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=device.type == "cuda")
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        batch[key] = tensor
    extra = [key for key in batch if key.startswith(IMAGE_PREFIX) and key not in camera_keys]
    for key in extra:
        tensor = batch[key]
        if not torch.is_tensor(tensor):
            continue
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=device.type == "cuda")
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        batch[key] = tensor


def _align_policy_images(batch: dict, rename_map: dict | None, image_features) -> None:
    for src, dst in (rename_map or {}).items():
        src_key = str(src)
        dst_key = str(dst)
        if src_key in batch and dst_key not in batch:
            batch[dst_key] = batch.pop(src_key)
        elif src_key in batch and src_key != dst_key:
            batch.pop(src_key, None)
    expected = [
        str(key)
        for key in (image_features or {})
        if str(key).startswith(IMAGE_PREFIX)
    ]
    if not expected or any(key in batch for key in expected):
        return
    leftover = sorted(
        key for key in batch if str(key).startswith(IMAGE_PREFIX)
    )
    for src, dst in zip(leftover, expected):
        if src != dst:
            batch[dst] = batch.pop(src)


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
    autocast=nullcontext,
    scaler=None,
) -> tuple[MetricsTracker, dict | None]:
    start_time = time.perf_counter()
    policy.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    with autocast():
        if sample_weights is not None:
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            float("inf"),
            error_if_nonfinite=False,
        )

    with lock if lock is not None else nullcontext():
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = float(loss.detach())
    grad_value = float(grad_norm)
    if grad_value == grad_value:  # not NaN
        train_metrics.grad_norm = grad_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    if torch.cuda.is_available():
        train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if output_dict:
        train_metrics.update_metrics(output_dict)
    return train_metrics, output_dict


def _setup_policy(cfg: TrainPipelineConfig, dataset, device: torch.device):
    if cfg.is_reward_model_training:
        from lerobot.rewards import make_reward_model

        logging.info("Creating reward model")
        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained."
            )
        return policy

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    if cfg.peft is not None:
        require_package("peft", extra="peft")
        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            policy = policy.wrap_with_peft(
                peft_cli_overrides=dataclasses.asdict(cfg.peft)
            )
    return policy


def _setup_processors(cfg: TrainPipelineConfig, policy, dataset, device: torch.device):
    active_cfg = cfg.trainable_config
    processor_pretrained_path = active_cfg.pretrained_path
    processor_kwargs: dict[str, Any] = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta
    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        preprocessor_overrides = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if cfg.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {
                "rename_map": cfg.rename_map
            }
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
        if not cfg.resume:
            preprocessor_overrides["normalizer_processor"]["stats"] = dataset.meta.stats
            postprocessor_overrides["unnormalizer_processor"]["stats"] = dataset.meta.stats
        if getattr(active_cfg, "use_relative_actions", False):
            preprocessor_overrides["relative_actions_processor"] = {
                "enabled": True,
                "exclude_joints": getattr(active_cfg, "relative_exclude_joints", []),
                "action_names": getattr(active_cfg, "action_feature_names", None),
            }
            postprocessor_overrides["absolute_actions_processor"] = {"enabled": True}
        processor_kwargs["preprocessor_overrides"] = preprocessor_overrides
        processor_kwargs["postprocessor_overrides"] = postprocessor_overrides
    if cfg.is_reward_model_training:
        return make_reward_pre_post_processors(cfg.reward_model, **processor_kwargs)
    return make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        pretrained_revision=getattr(cfg.policy, "pretrained_revision", None),
        **processor_kwargs,
    )


def _make_dataloader(cfg: TrainPipelineConfig, dataset, device: torch.device, step: int):
    if cfg.dataset.streaming:
        kwargs = {
            "num_workers": cfg.num_workers,
            "pin_memory": device.type == "cuda",
            "drop_last": False,
            "collate_fn": variable_camera_collate,
            "batch_size": cfg.batch_size,
            "shuffle": True,
            **_dataloader_worker_kwargs(cfg),
        }
        if cfg.num_workers > 0:
            kwargs["worker_init_fn"] = _worker_init
        loader = torch.utils.data.DataLoader(dataset, **kwargs)
        return loader, cycle(loader)

    tables = getattr(dataset, "frame_tables", None) or {}
    if not tables:
        raise RuntimeError("episode frame tables are empty; cannot stream the dataset")

    workers = int(cfg.num_workers)
    prefetch = max(int(cfg.prefetch_factor), 4) if workers > 0 else None
    loader = torch.utils.data.DataLoader(
        EpisodeStreamDataset(
            dataset,
            batch_size=cfg.batch_size,
            seed=cfg.seed if cfg.seed is not None else 0,
        ),
        batch_size=None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init if workers > 0 else None,
        prefetch_factor=prefetch,
        persistent_workers=cfg.persistent_workers and workers > 0,
        multiprocessing_context=cfg.dataloader_multiprocessing_context
        if workers > 0
        else None,
    )
    logging.info(
        "Episode-stream dataloader · %s episodes · window %s frames · "
        "batch %s · %s workers · prefetch_factor=%s batches",
        len(tables),
        max(PREFETCH_FRAMES, int(cfg.batch_size)),
        cfg.batch_size,
        workers,
        prefetch,
    )
    if cfg.resume and step > 0:
        logging.info("Episode-stream dataloader does not restore sample order on resume")
    return loader, cycle(loader)


def train(cfg: TrainPipelineConfig) -> None:
    if cfg.job.is_remote:
        return submit_to_hf(cfg)

    cfg.validate()
    init_logging()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.trainable_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device} is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        device = torch.device("cuda", torch.cuda.current_device())
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logging.info("Creating dataset")
    dataset, eval_dataset = make_train_eval_datasets(cfg)
    drop_n_last = int(getattr(cfg.trainable_config, "drop_n_last_frames", 0) or 0)
    dataset = wrap_dataset(
        dataset, num_workers=cfg.num_workers, drop_n_last_frames=drop_n_last
    )
    if eval_dataset is not None:
        eval_dataset = wrap_dataset(
            eval_dataset, cache_frames=False, num_workers=cfg.num_workers
        )

    policy = _setup_policy(cfg, dataset, device)
    logging.info(
        "Init from %s · images %s · rename %s",
        getattr(cfg.policy, "pretrained_path", None) or cfg.policy.type,
        list(getattr(policy.config, "image_features", {}) or {}),
        dict(cfg.rename_map or {}),
    )
    policy.to(device)
    if device.type == "cuda":
        policy.float()
    preprocessor, postprocessor = _setup_processors(cfg, policy, dataset, device)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    env_preprocessor = None
    env_postprocessor = None
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
        logging.info("Creating environment processors")
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"Effective batch size: {cfg.batch_size}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    dataloader, dl_iter = _make_dataloader(cfg, dataset, device, step)
    eval_dataloader = None
    if eval_dataset is not None:
        eval_ds = eval_dataset
        if cfg.max_eval_samples > 0 and hasattr(eval_dataset, "hf_dataset"):
            task_arr = eval_dataset.hf_dataset.data.column("task_index").to_numpy()
            unique_tasks = sorted(set(task_arr.tolist()))
            per_task = max(1, cfg.max_eval_samples // len(unique_tasks))
            selected: list[int] = []
            for task in unique_tasks:
                frames = (task_arr == task).nonzero()[0][:per_task]
                selected.extend(frames.tolist())
            eval_ds = torch.utils.data.Subset(eval_dataset, selected)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=variable_camera_collate,
            **_dataloader_worker_kwargs(cfg),
        )

    use_amp, amp_dtype, needs_scaler = resolve_amp(
        device, bool(getattr(cfg.trainable_config, "use_amp", False))
    )
    scaler = make_scaler(needs_scaler, device)
    if use_amp:
        autocast = lambda: torch.autocast(device.type, dtype=amp_dtype)
        logging.info(f"AMP on · {amp_dtype} · scaler={needs_scaler}")
    else:
        autocast = nullcontext
        logging.info("AMP off")

    vision_cache = None
    cache_on, cache_bytes = _vision_cache_config()
    if getattr(cfg.trainable_config, "freeze_vision_encoder", False) and cache_on:
        vision_cache = _install_vision_cache(policy, cache_bytes)
        if vision_cache is not None:
            logging.info(
                f"Frozen vision embedding cache on · {cache_bytes / 1024**3:.1f} GB CPU"
            )
    elif getattr(cfg.trainable_config, "freeze_vision_encoder", False):
        logging.info("Frozen vision embedding cache off")

    policy.train()
    camera_keys = list(dataset.meta.camera_keys)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f", reduction="mean"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f", reduction="max"),
        "dataloading_s": AverageMeter("data_s", ":.3f", reduction="max"),
        "samples_per_s": AverageMeter("smp/s", ":.0f"),
    }
    if torch.cuda.is_available():
        train_metrics["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f", reduction="max")

    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    progbar = tqdm(
        total=cfg.steps - step,
        desc="Training",
        unit="step",
        disable=inside_slurm(),
        position=0,
        leave=True,
    )
    logging.info(
        f"Start offline training on a fixed dataset, with effective batch size: {cfg.batch_size}"
    )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        _move_images(batch, device, camera_keys)
        _align_policy_images(
            batch,
            cfg.rename_map,
            getattr(policy.config, "image_features", None),
        )
        if vision_cache is not None:
            vision_cache.set_batch(batch)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, _ = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
            autocast=autocast,
            scaler=scaler,
        )

        step += 1
        progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = should_save_checkpoint(step, cfg.save_freq, cfg.steps)
        is_env_eval_step = cfg.env_eval_freq > 0 and step % cfg.env_eval_freq == 0
        is_eval_step = (
            cfg.eval_steps > 0 and eval_dataloader is not None and step % cfg.eval_steps == 0
        )

        if is_log_step:
            step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
            if step_time > 0:
                train_tracker.samples_per_s = cfg.batch_size / step_time
            logging.info(train_tracker)
            print(
                f"step:{step} smpl:{int(train_tracker.samples)} epch:{train_tracker.epochs:.2f} "
                f"loss:{float(train_tracker.loss.avg):.4f} "
                f"grdn:{float(train_tracker.grad_norm.avg):.3f} "
                f"lr:{float(train_tracker.lr.avg):g}",
                flush=True,
            )
            if vision_cache is not None:
                total = vision_cache.hits + vision_cache.misses
                rate = vision_cache.hits / total if total else 0.0
                logging.info(
                    f"vision_cache hit={vision_cache.hits} miss={vision_cache.misses} "
                    f"rate={rate:.2f} keys={len(vision_cache.cache)}"
                )
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if sample_weighter is not None:
                    weighter_stats = sample_weighter.get_stats()
                    wandb_log_dict.update(
                        {f"sample_weighting/{key}": value for key, value in weighter_stats.items()}
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_eval_step:
            policy.eval()
            eval_loss_sum = 0.0
            n_eval_batches = 0
            with torch.no_grad(), autocast():
                for eval_batch in eval_dataloader:
                    _move_images(eval_batch, device, camera_keys)
                    _align_policy_images(
                        eval_batch,
                        cfg.rename_map,
                        getattr(policy.config, "image_features", None),
                    )
                    eval_batch = preprocessor(eval_batch)
                    loss, _ = policy.forward(eval_batch)
                    eval_loss_sum += float(loss.detach())
                    n_eval_batches += 1
            eval_loss = eval_loss_sum / max(n_eval_batches, 1)
            policy.train()
            logging.info(f"step {step}: eval_loss={eval_loss:.4f}")
            if wandb_logger:
                wandb_logger.log_dict({"eval_loss": eval_loss}, step=step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                num_processes=1,
                batch_size=cfg.batch_size,
            )
            update_last_checkpoint(checkpoint_dir)
            if cfg.save_checkpoint_to_hub:
                push_checkpoint_to_hub(
                    checkpoint_dir,
                    cfg.policy.repo_id,
                    private=cfg.policy.private,
                )
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_env_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with _make_eval_envs(cfg) as eval_env, torch.no_grad(), autocast():
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            aggregated = eval_info["overall"]
            for suite, suite_info in eval_info.items():
                logging.info("Suite %s aggregated: %s", suite, suite_info)
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size,
                dataset.num_frames,
                dataset.num_episodes,
                eval_metrics,
                initial_step=step,
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_logger.log_dict({**eval_tracker.to_dict(), **eval_info}, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

    progbar.close()
    logging.info("End of training")
    try:
        iterator = getattr(dataloader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    except Exception:
        pass

    active_cfg = cfg.trainable_config
    if getattr(active_cfg, "push_to_hub", False):
        if not cfg.is_reward_model_training and cfg.policy.use_peft:
            policy.push_model_to_hub(cfg, peft_model=policy, dataset_meta=dataset.meta)
        else:
            policy.push_model_to_hub(cfg, dataset_meta=dataset.meta)
        preprocessor.push_to_hub(active_cfg.repo_id)
        postprocessor.push_to_hub(active_cfg.repo_id)
