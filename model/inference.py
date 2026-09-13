from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, SmolVLMForConditionalGeneration

_SIM_ROOT = Path(__file__).resolve().parents[1] / "sim"
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

from core.config import LLM_LOG_PATH, resolve_source, stored_source

# ======Settings=========
MODEL_ID = "lerobot/smolvla_base"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
STATS_FILE = "policy_preprocessor_step_5_normalizer_processor.safetensors"
STATS_KEY = "so100"
CAMERA_ORDER = ("front", "wrist", "overview", "top", "left", "right")
DEFAULT_CAMERAS = ("front", "wrist")
MAX_CAMERAS = 5
IMAGE_SIZE = 512
TOKENIZER_MAX_LENGTH = 48
CHUNK_SIZE = 50
ACTION_DIM = 6
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
NUM_VLM_LAYERS = 16
SELF_ATTN_EVERY_N = 2
EXPERT_WIDTH_MULTIPLIER = 0.75
NUM_DENOISE_STEPS = 10
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0
NORM_EPS = 1e-8
USE_CUDA_GRAPHS = True
# ======Settings=========


def _tensor_from_stats(stats: dict, names: tuple[str, ...]) -> Tensor | None:
    for name in names:
        if name in stats:
            return stats[name]
    return None


def _load_weight_file(source: str) -> dict:
    folder = resolve_source(source)
    if isinstance(folder, Path) and folder.is_dir():
        path = folder / "model.safetensors"
        if not path.is_file():
            raise RuntimeError(f"no model.safetensors in {folder}")
        return load_file(path)
    return load_file(hf_hub_download(source, "model.safetensors"))


def _load_norm_stats(source: str) -> dict:
    folder = resolve_source(source)
    if isinstance(folder, Path) and folder.is_dir():
        tensors = {}
        for path in sorted(folder.glob("*.safetensors")):
            if path.name == "model.safetensors":
                continue
            tensors.update(load_file(path))
        if tensors:
            return tensors
    return load_file(hf_hub_download(MODEL_ID, STATS_FILE))


def _normalize_camera_names(value) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= MAX_CAMERAS:
            break
    return names


def _cameras_from_rename(rename: dict) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for src, dst in (rename or {}).items():
        src_name = str(src).rsplit(".", 1)[-1]
        dst_name = str(dst).rsplit(".", 1)[-1]
        index = 10_000
        if dst_name.startswith("camera"):
            try:
                index = int(dst_name[6:])
            except ValueError:
                pass
        pairs.append((index, src_name))
    pairs.sort()
    return _normalize_camera_names([name for _, name in pairs])


def _policy_meta(source: str) -> tuple[int, list[str]]:
    empty = 0
    cameras: list[str] = []
    folder = resolve_source(source)
    if not (isinstance(folder, Path) and folder.is_dir()):
        return empty, cameras
    cfg_path = folder / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            cfg = {}
        empty = max(0, int(cfg.get("empty_cameras") or 0))
    for name in ("policy_preprocessor.json", "train_config.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rename = {}
        if name == "train_config.json":
            rename = dict(data.get("rename_map") or {})
            policy = data.get("policy") or {}
            if not empty:
                empty = max(0, int(policy.get("empty_cameras") or 0))
        else:
            for step in data.get("steps") or []:
                cfg = (step or {}).get("config") or {}
                if cfg.get("rename_map"):
                    rename = dict(cfg["rename_map"])
                    break
        if rename:
            cameras = _cameras_from_rename(rename)
            if cameras:
                break
    return empty, cameras


def _inference_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def _configure_backends(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def _intermediate_size(hidden_dim: int) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(4 * hidden_dim)
    return 256 * ((hidden_dim + 255) // 256)


def _rope_cis(positions: Tensor, inv_freq: Tensor) -> tuple[Tensor, Tensor]:
    radians = positions[..., None].to(inv_freq.dtype) * inv_freq
    radians = radians.unsqueeze(-2)
    return radians.cos(), radians.sin()


def _apply_rope_cis(states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    left, right = states.chunk(2, dim=-1)
    if cos.dtype != states.dtype:
        cos = cos.to(dtype=states.dtype)
        sin = sin.to(dtype=states.dtype)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


def _resize_pad(image: Tensor, width: int, height: int) -> Tensor:
    _, _, current_h, current_w = image.shape
    ratio = max(current_w / width, current_h / height)
    new_h = max(1, int(current_h / ratio))
    new_w = max(1, int(current_w / ratio))
    resized = F.interpolate(
        image, size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return F.pad(resized, (max(0, width - new_w), 0, max(0, height - new_h), 0))


def _attention_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    cumulative = torch.cumsum(att_masks, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    return causal & (pad_masks[:, None, :] * pad_masks[:, :, None])


def _sdpa(mask: Tensor, query: Tensor, key: Tensor, value: Tensor, head_dim: int) -> Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask[:, None, :, :],
        scale=head_dim**-0.5,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    batch, heads, length, _ = output.shape
    return output.transpose(1, 2).reshape(batch, length, heads * head_dim)


def _fuse_qkv(layer: nn.Module) -> None:
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj
    if q_proj.in_features != k_proj.in_features:
        return
    fused = nn.Linear(
        q_proj.in_features,
        q_proj.out_features + k_proj.out_features + v_proj.out_features,
        bias=q_proj.bias is not None,
        device=q_proj.weight.device,
        dtype=q_proj.weight.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        if fused.bias is not None:
            fused.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))
    layer.self_attn.qkv_proj = fused
    layer.self_attn.q_out = q_proj.out_features
    layer.self_attn.k_out = k_proj.out_features


class SmolVLMWithExpert(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(VLM_ID)
        config.text_config.num_hidden_layers = NUM_VLM_LAYERS
        self.vlm = SmolVLMForConditionalGeneration(config)
        expert_config = copy.deepcopy(config.text_config)
        hidden = int(expert_config.hidden_size * EXPERT_WIDTH_MULTIPLIER)
        expert_config.hidden_size = hidden
        expert_config.intermediate_size = _intermediate_size(hidden)
        expert_config.num_hidden_layers = NUM_VLM_LAYERS
        self.lm_expert = AutoModel.from_config(expert_config)
        kv_in = config.text_config.num_key_value_heads * config.text_config.head_dim
        kv_out = expert_config.num_key_value_heads * expert_config.head_dim
        for index, layer in enumerate(self.lm_expert.layers):
            if index % SELF_ATTN_EVERY_N == 0:
                continue
            layer.self_attn.k_proj = nn.Linear(
                kv_in, kv_out, bias=expert_config.attention_bias
            )
            layer.self_attn.v_proj = nn.Linear(
                kv_in, kv_out, bias=expert_config.attention_bias
            )
        self.lm_expert.embed_tokens = None
        self.num_q_heads = config.text_config.num_attention_heads
        self.num_kv_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.expert_hidden_size = hidden
        self.hidden_size = config.text_config.hidden_size
        half = self.head_dim // 2
        exponents = (2.0 / self.head_dim) * torch.arange(half, dtype=torch.float32, device=device)
        self.register_buffer(
            "rope_inv_freq",
            (10_000.0 ** exponents).reciprocal(),
            persistent=False,
        )
        self.to(device)
        self._vlm = self.vlm.model
        self.vlm_layers = list(self._vlm.text_model.layers)
        self.expert_layers = list(self.lm_expert.layers)
        self._embed_tokens = self._vlm.text_model.get_input_embeddings()
        self._image_tokens: int | None = None

    def get_vlm(self):
        return self._vlm

    def embed_images(self, stacked: Tensor) -> Tensor:
        stacked = stacked.to(
            dtype=self._vlm.vision_model.dtype,
            memory_format=torch.channels_last,
        )
        hidden = self._vlm.vision_model(pixel_values=stacked).last_hidden_state
        emb = self._vlm.connector(hidden)
        self._image_tokens = emb.shape[1]
        return emb

    def embed_language(self, tokens: Tensor) -> Tensor:
        return self._embed_tokens(tokens)

    def _project(self, layer, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = layer.input_layernorm(hidden)
        attn = layer.self_attn
        hidden = hidden.to(dtype=attn.qkv_proj.weight.dtype)
        query, key, value = attn.qkv_proj(hidden).split(
            (attn.q_out, attn.k_out, attn.k_out), dim=-1
        )
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        return query.view(shape), key.view(shape), value.view(shape)

    def _residual(self, layer, hidden: Tensor, attn: Tensor) -> Tensor:
        attn = attn.to(dtype=layer.self_attn.o_proj.weight.dtype)
        output = layer.self_attn.o_proj(attn) + hidden
        return output + layer.mlp(layer.post_attention_layernorm(output))

    def fuse_qkv(self) -> None:
        for layer in self.vlm_layers:
            _fuse_qkv(layer)
        for index, layer in enumerate(self.expert_layers):
            if index % SELF_ATTN_EVERY_N == 0:
                _fuse_qkv(layer)

    def forward_prefix(
        self,
        attention_mask: Tensor,
        prefix: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        cache_k: list[Tensor] = []
        cache_v: list[Tensor] = []
        hidden = prefix
        for layer in self.vlm_layers:
            query, key, value = self._project(layer, hidden)
            query = _apply_rope_cis(query, cos, sin)
            key = _apply_rope_cis(key, cos, sin)
            cache_k.append(key)
            cache_v.append(value)
            attn = _sdpa(attention_mask, query, key, value, self.head_dim)
            hidden = self._residual(layer, hidden, attn)
        return cache_k, cache_v

    def project_cross_kv(
        self,
        cache_k: list[Tensor],
        cache_v: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        batch = cache_k[0].shape[0]
        expert_k: list[Tensor] = []
        expert_v: list[Tensor] = []
        for layer_idx in range(1, NUM_VLM_LAYERS, SELF_ATTN_EVERY_N):
            layer = self.expert_layers[layer_idx]
            prefix_len = cache_k[layer_idx].shape[1]
            key_flat = cache_k[layer_idx].reshape(batch, prefix_len, -1)
            value_flat = cache_v[layer_idx].reshape(batch, prefix_len, -1)
            expert_k.append(
                layer.self_attn.k_proj(key_flat).view(
                    batch, prefix_len, -1, layer.self_attn.head_dim
                )
            )
            expert_v.append(
                layer.self_attn.v_proj(value_flat).view(
                    batch, prefix_len, -1, layer.self_attn.head_dim
                )
            )
        return expert_k, expert_v

    def forward_suffix(
        self,
        self_mask: Tensor,
        cross_mask: Tensor,
        suffix: Tensor,
        cache_k: list[Tensor],
        cache_v: list[Tensor],
        expert_k: list[Tensor],
        expert_v: list[Tensor],
        self_cos: Tensor,
        self_sin: Tensor,
        cross_cos: Tensor,
        cross_sin: Tensor,
    ) -> Tensor:
        hidden = suffix
        cross_idx = 0
        for layer_idx in range(0, NUM_VLM_LAYERS, SELF_ATTN_EVERY_N):
            layer = self.expert_layers[layer_idx]
            query, key, value = self._project(layer, hidden)
            query = _apply_rope_cis(query, self_cos, self_sin)
            key = _apply_rope_cis(key, self_cos, self_sin)
            attn = _sdpa(
                self_mask,
                query,
                torch.cat([cache_k[layer_idx], key], dim=1),
                torch.cat([cache_v[layer_idx], value], dim=1),
                self.head_dim,
            )
            hidden = self._residual(layer, hidden, attn)

            layer = self.expert_layers[layer_idx + 1]
            normed = layer.input_layernorm(hidden).to(dtype=layer.self_attn.q_proj.weight.dtype)
            query = layer.self_attn.q_proj(normed).view(
                *normed.shape[:-1], -1, layer.self_attn.head_dim
            )
            query = _apply_rope_cis(query, cross_cos, cross_sin)
            attn = _sdpa(
                cross_mask,
                query,
                expert_k[cross_idx],
                expert_v[cross_idx],
                self.head_dim,
            )
            hidden = self._residual(layer, hidden, attn)
            cross_idx += 1
        return self.lm_expert.norm(hidden)


class FlowPolicy(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.vlm_with_expert = SmolVLMWithExpert(device)
        hidden = self.vlm_with_expert.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size
        self.state_proj = nn.Linear(MAX_STATE_DIM, hidden)
        self.action_in_proj = nn.Linear(MAX_ACTION_DIM, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, MAX_ACTION_DIM)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        half = expert_hidden // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=device)
        self.register_buffer(
            "time_scale",
            (1.0 / (MIN_PERIOD * (MAX_PERIOD / MIN_PERIOD) ** fraction)) * (2 * math.pi),
            persistent=False,
        )
        self._image_scale = math.sqrt(hidden)
        self.empty_cameras = 0
        self._lang_cache: tuple[Tensor, Tensor] | None = None
        self._dummy_image: Tensor | None = None
        self._prefix_mask: Tensor | None = None
        self._self_mask: Tensor | None = None
        self._cross_mask: Tensor | None = None
        self._prefix_cos: Tensor | None = None
        self._prefix_sin: Tensor | None = None
        self._self_cos: Tensor | None = None
        self._self_sin: Tensor | None = None
        self._cross_cos: Tensor | None = None
        self._cross_sin: Tensor | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graphs_ok = True
        self._g_current: Tensor | None = None
        self._g_time: Tensor | None = None
        self._g_vel: Tensor | None = None
        self._g_cache_k: list[Tensor] | None = None
        self._g_cache_v: list[Tensor] | None = None
        self._g_expert_k: list[Tensor] | None = None
        self._g_expert_v: list[Tensor] | None = None
        self.to(device)

    def reset_runtime(self) -> None:
        self._graph = None
        self._graphs_ok = True
        self._self_mask = None
        self._cross_mask = None
        self._self_cos = None
        self._self_sin = None
        self._cross_cos = None
        self._cross_sin = None
        self._g_current = None
        self._g_time = None
        self._g_vel = None
        self._g_cache_k = None
        self._g_cache_v = None
        self._g_expert_k = None
        self._g_expert_v = None
        self._dummy_image = None
        self._lang_cache = None

    def _cached_lang(self, lang_tokens: Tensor) -> Tensor:
        if self._lang_cache is None or self._lang_cache[0] is not lang_tokens:
            lang = self.vlm_with_expert.embed_language(lang_tokens)
            lang = lang * self._image_scale
            self._lang_cache = (lang_tokens, lang)
        return self._lang_cache[1]

    def embed_prefix(
        self,
        images: Tensor,
        n_real: int,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        n_real = max(0, int(n_real))
        n_empty = max(0, int(self.empty_cameras))
        if n_real <= 0:
            raise RuntimeError("inference requires at least one camera frame")
        if images.ndim == 5:
            batch = images.shape[0]
            real = images[:, :n_real].reshape(batch * n_real, *images.shape[2:])
        else:
            batch = 1
            real = images[:n_real]
        real_emb = self.vlm_with_expert.embed_images(real)
        tokens = real_emb.shape[1]
        real_emb = real_emb.view(batch, n_real, tokens, -1)
        slots = [real_emb]
        if n_empty:
            dummy = torch.ones(
                batch * n_empty,
                3,
                real.shape[-2],
                real.shape[-1],
                device=real.device,
                dtype=real.dtype,
            )
            dummy.mul_(-1.0)
            dummy = dummy.to(memory_format=torch.channels_last)
            empty_emb = self.vlm_with_expert.embed_images(dummy)
            slots.append(empty_emb.view(batch, n_empty, tokens, -1))
        n_slots = n_real + n_empty
        image_emb = torch.cat(slots, dim=1).reshape(batch, n_slots * tokens, -1)
        image_emb = image_emb * self._image_scale
        if n_empty:
            image_pad = torch.cat(
                (
                    torch.ones(batch, n_real * tokens, dtype=torch.bool, device=state.device),
                    torch.zeros(batch, n_empty * tokens, dtype=torch.bool, device=state.device),
                ),
                dim=1,
            )
        else:
            image_pad = torch.ones(
                batch, n_slots * tokens, dtype=torch.bool, device=state.device
            )
        lang = self._cached_lang(lang_tokens)
        if lang.shape[0] != batch:
            if lang.shape[0] != 1:
                self._lang_cache = None
                lang = self._cached_lang(lang_tokens)
            if lang.shape[0] != batch:
                if lang.shape[0] != 1:
                    raise RuntimeError(
                        f"language batch {lang.shape[0]} != image batch {batch}"
                    )
                lang = lang.expand(batch, -1, -1)
        if lang_masks.shape[0] != batch:
            if lang_masks.shape[0] != 1:
                raise RuntimeError(
                    f"language mask batch {lang_masks.shape[0]} != image batch {batch}"
                )
            lang_masks = lang_masks.expand(batch, -1)
        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None]
        embeds = torch.cat([image_emb, lang, state_emb], dim=1)
        pad = torch.cat(
            [
                image_pad,
                lang_masks,
                torch.ones(batch, state_emb.shape[1], dtype=torch.bool, device=state.device),
            ],
            dim=1,
        )
        att = torch.zeros(batch, embeds.shape[1], dtype=torch.bool, device=state.device)
        att[:, -state_emb.shape[1] :] = True
        return embeds, pad, att

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> Tensor:
        action_emb = self.action_in_proj(noisy_actions)
        angle = self.time_scale[None, :] * timestep.to(self.time_scale.dtype)[:, None]
        time_emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1).to(dtype=action_emb.dtype)
        fused = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=2)
        return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(fused)))

    def _denoise_step(
        self,
        current: Tensor,
        time_value: Tensor,
        cache_k: list[Tensor],
        cache_v: list[Tensor],
        expert_k: list[Tensor],
        expert_v: list[Tensor],
        self_mask: Tensor | None = None,
        cross_mask: Tensor | None = None,
        self_cos: Tensor | None = None,
        self_sin: Tensor | None = None,
        cross_cos: Tensor | None = None,
        cross_sin: Tensor | None = None,
    ) -> Tensor:
        suffix = self.embed_suffix(current, time_value).to(
            dtype=self.vlm_with_expert.lm_expert.norm.weight.dtype
        )
        hidden = self.vlm_with_expert.forward_suffix(
            self._self_mask if self_mask is None else self_mask,
            self._cross_mask if cross_mask is None else cross_mask,
            suffix,
            cache_k,
            cache_v,
            expert_k,
            expert_v,
            self._self_cos if self_cos is None else self_cos,
            self._self_sin if self_sin is None else self_sin,
            self._cross_cos if cross_cos is None else cross_cos,
            self._cross_sin if cross_sin is None else cross_sin,
        )
        return self.action_out_proj(hidden[:, -CHUNK_SIZE:].float())

    def _suffix_attn(
        self, prefix_pad: Tensor, prefix_att: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = prefix_pad.shape[0]
        device = prefix_pad.device
        suffix_pad = torch.ones(batch, CHUNK_SIZE, dtype=torch.bool, device=device)
        suffix_att = torch.ones(batch, CHUNK_SIZE, dtype=torch.bool, device=device)
        suffix_mask = _attention_masks(suffix_pad, suffix_att)
        self_mask = torch.cat(
            [prefix_pad[:, None, :].expand(batch, CHUNK_SIZE, -1), suffix_mask],
            dim=2,
        )
        cross_mask = prefix_pad[:, None, :].expand(batch, CHUNK_SIZE, -1)
        prefix_len_valid = prefix_pad.sum(dim=-1)
        position_ids = prefix_len_valid[:, None] + torch.arange(CHUNK_SIZE, device=device)
        rel_pos = position_ids - position_ids.min(dim=1, keepdim=True).values
        inv_freq = self.vlm_with_expert.rope_inv_freq
        rope_dtype = self.vlm_with_expert.lm_expert.norm.weight.dtype
        self_cos, self_sin = _rope_cis(position_ids, inv_freq)
        cross_cos, cross_sin = _rope_cis(rel_pos, inv_freq)
        return (
            self_mask,
            cross_mask,
            self_cos.to(dtype=rope_dtype),
            self_sin.to(dtype=rope_dtype),
            cross_cos.to(dtype=rope_dtype),
            cross_sin.to(dtype=rope_dtype),
        )

    def _ensure_static_masks(self, prefix_pad: Tensor, prefix_att: Tensor) -> None:
        self_mask, cross_mask, self_cos, self_sin, cross_cos, cross_sin = self._suffix_attn(
            prefix_pad, prefix_att
        )
        if self._self_mask is not None and self._self_mask.shape != self_mask.shape:
            self._graph = None
            self._g_current = None
            self._g_cache_k = None
            self._g_cache_v = None
            self._g_expert_k = None
            self._g_expert_v = None
            self._self_mask = None
        if self._self_mask is None:
            self._self_mask = self_mask.contiguous()
            self._cross_mask = cross_mask.contiguous()
            self._self_cos = self_cos.contiguous()
            self._self_sin = self_sin.contiguous()
            self._cross_cos = cross_cos.contiguous()
            self._cross_sin = cross_sin.contiguous()
            return
        self._self_mask.copy_(self_mask)
        self._cross_mask.copy_(cross_mask)
        self._self_cos.copy_(self_cos)
        self._self_sin.copy_(self_sin)
        self._cross_cos.copy_(cross_cos)
        self._cross_sin.copy_(cross_sin)

    def _capture_denoise(self, current: Tensor, time_value: Tensor) -> None:
        if not USE_CUDA_GRAPHS or current.device.type != "cuda":
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._g_vel = self._denoise_step(
                    self._g_current,
                    self._g_time,
                    self._g_cache_k,
                    self._g_cache_v,
                    self._g_expert_k,
                    self._g_expert_v,
                )
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._g_vel = self._denoise_step(
                self._g_current,
                self._g_time,
                self._g_cache_k,
                self._g_cache_v,
                self._g_expert_k,
                self._g_expert_v,
            )

    def sample_actions(
        self,
        images: Tensor,
        n_real: int,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        num_steps: int,
    ) -> Tensor:
        batch = state.shape[0]
        device = state.device
        prefix, prefix_pad, prefix_att = self.embed_prefix(
            images, n_real, lang_tokens, lang_masks, state
        )
        self._ensure_static_masks(prefix_pad, prefix_att)
        prefix_mask = _attention_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_cos, prefix_sin = _rope_cis(prefix_pos, self.vlm_with_expert.rope_inv_freq)
        prefix_cos = prefix_cos.to(dtype=prefix.dtype)
        prefix_sin = prefix_sin.to(dtype=prefix.dtype)
        cache_k, cache_v = self.vlm_with_expert.forward_prefix(
            prefix_mask, prefix, prefix_cos, prefix_sin
        )
        expert_k, expert_v = self.vlm_with_expert.project_cross_kv(cache_k, cache_v)
        dt = -1.0 / num_steps
        current = torch.randn(
            batch, CHUNK_SIZE, MAX_ACTION_DIM, dtype=torch.float32, device=device
        )
        use_graphs = (
            USE_CUDA_GRAPHS
            and device.type == "cuda"
            and not any(param.requires_grad for param in self.parameters())
        )
        graph_batch = (
            int(self._g_current.shape[0]) if self._g_current is not None else -1
        )
        if (
            use_graphs
            and self._graph is not None
            and graph_batch == batch
        ):
            for dst, src in zip(self._g_cache_k, cache_k, strict=True):
                dst.copy_(src)
            for dst, src in zip(self._g_cache_v, cache_v, strict=True):
                dst.copy_(src)
            for dst, src in zip(self._g_expert_k, expert_k, strict=True):
                dst.copy_(src)
            for dst, src in zip(self._g_expert_v, expert_v, strict=True):
                dst.copy_(src)
            self._g_current.copy_(current)
            for step in range(num_steps):
                self._g_time.fill_(1.0 + step * dt)
                self._graph.replay()
                self._g_current.add_(self._g_vel, alpha=dt)
            return self._g_current.clone()

        if use_graphs and self._graph is None:
            self._g_current = current
            self._g_time = torch.ones(batch, dtype=torch.float32, device=device)
            self._g_cache_k = cache_k
            self._g_cache_v = cache_v
            self._g_expert_k = expert_k
            self._g_expert_v = expert_v
            try:
                self._capture_denoise(current, self._g_time)
            except Exception:
                self._graph = None
            if self._graph is not None:
                for step in range(num_steps):
                    self._g_time.fill_(1.0 + step * dt)
                    self._graph.replay()
                    self._g_current.add_(self._g_vel, alpha=dt)
                return self._g_current.clone()

        for step in range(num_steps):
            time_value = torch.full(
                (batch,), 1.0 + step * dt, dtype=torch.float32, device=device
            )
            velocity = self._denoise_step(
                current, time_value, cache_k, cache_v, expert_k, expert_v
            )
            current = current + dt * velocity
        return current

    def prefix_caches(
        self,
        images: Tensor,
        n_real: int,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        bind_static: bool = True,
        vlm_no_grad: bool = False,
    ) -> tuple:
        context = torch.no_grad() if vlm_no_grad else torch.enable_grad()
        with context:
            prefix, prefix_pad, prefix_att = self.embed_prefix(
                images, n_real, lang_tokens, lang_masks, state
            )
            suffix = None
            if bind_static:
                self._ensure_static_masks(prefix_pad, prefix_att)
            else:
                suffix = self._suffix_attn(prefix_pad, prefix_att)
            prefix_mask = _attention_masks(prefix_pad, prefix_att)
            prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
            prefix_cos, prefix_sin = _rope_cis(prefix_pos, self.vlm_with_expert.rope_inv_freq)
            prefix_cos = prefix_cos.to(dtype=prefix.dtype)
            prefix_sin = prefix_sin.to(dtype=prefix.dtype)
            cache_k, cache_v = self.vlm_with_expert.forward_prefix(
                prefix_mask, prefix, prefix_cos, prefix_sin
            )
        if vlm_no_grad:
            cache_k = [item.detach() for item in cache_k]
            cache_v = [item.detach() for item in cache_v]
        expert_k, expert_v = self.vlm_with_expert.project_cross_kv(cache_k, cache_v)
        if suffix is None:
            return cache_k, cache_v, expert_k, expert_v
        return cache_k, cache_v, expert_k, expert_v, suffix

    def _copy_graph_caches(
        self,
        cache_k: list[Tensor],
        cache_v: list[Tensor],
        expert_k: list[Tensor],
        expert_v: list[Tensor],
    ) -> None:
        for dst, src in zip(self._g_cache_k, cache_k, strict=True):
            dst.copy_(src)
        for dst, src in zip(self._g_cache_v, cache_v, strict=True):
            dst.copy_(src)
        for dst, src in zip(self._g_expert_k, expert_k, strict=True):
            dst.copy_(src)
        for dst, src in zip(self._g_expert_v, expert_v, strict=True):
            dst.copy_(src)

    def _pad_to_graph_batch(
        self,
        images: Tensor,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        real_batch = int(state.shape[0])
        target = int(self._g_current.shape[0]) if self._g_current is not None else real_batch
        if real_batch >= target:
            return images, lang_tokens, lang_masks, state, real_batch
        pad = target - real_batch
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = torch.cat([images, images[:1].repeat(pad, *([1] * (images.ndim - 1)))], dim=0)
        state = torch.cat([state, state[:1].repeat(pad, *([1] * (state.ndim - 1)))], dim=0)
        if lang_tokens.shape[0] > 1:
            lang_tokens = torch.cat(
                [lang_tokens, lang_tokens[:1].repeat(pad, *([1] * (lang_tokens.ndim - 1)))],
                dim=0,
            )
            lang_masks = torch.cat(
                [lang_masks, lang_masks[:1].repeat(pad, *([1] * (lang_masks.ndim - 1)))],
                dim=0,
            )
        return images, lang_tokens, lang_masks, state, real_batch

    def sample_actions_sde(
        self,
        images: Tensor,
        n_real: int,
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        num_steps: int,
        noise_level: float,
        sde_steps: set[int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
        from grpo.flow_sde import sde_step

        device = state.device
        dt = -1.0 / num_steps
        use_graphs = (
            USE_CUDA_GRAPHS
            and self._graphs_ok
            and device.type == "cuda"
            and torch.is_inference_mode_enabled()
        )
        if (
            use_graphs
            and self._g_current is not None
            and int(state.shape[0]) > int(self._g_current.shape[0])
        ):
            self._graph = None
            self._g_current = None
        if use_graphs and self._g_current is not None:
            images, lang_tokens, lang_masks, state, real_batch = self._pad_to_graph_batch(
                images, lang_tokens, lang_masks, state
            )
        else:
            real_batch = int(state.shape[0])
        batch = int(state.shape[0])
        current = torch.randn(
            batch, CHUNK_SIZE, MAX_ACTION_DIM, dtype=torch.float32, device=device
        )
        cache_k, cache_v, expert_k, expert_v = self.prefix_caches(
            images, n_real, lang_tokens, lang_masks, state, bind_static=True
        )
        traj = [current.clone()]
        logps = []
        mask = []

        def _advance(velocity: Tensor, step: int, x: Tensor) -> Tensor:
            tau = torch.full((batch,), 1.0 + step * dt, dtype=torch.float32, device=device)
            if step in sde_steps:
                result = sde_step(x, velocity, tau, dt, noise_level)
                nxt = result.x_next
                if not torch.isfinite(nxt).all():
                    nxt = torch.nan_to_num(nxt, nan=0.0, posinf=0.0, neginf=0.0)
                logps.append(result.log_prob)
                mask.append(True)
                return nxt
            logps.append(torch.zeros(batch, device=device))
            mask.append(False)
            return x + dt * velocity

        if (
            use_graphs
            and self._graph is not None
            and self._g_current is not None
            and self._g_current.shape == current.shape
        ):
            self._copy_graph_caches(cache_k, cache_v, expert_k, expert_v)
            self._g_current.copy_(current)
            for step in range(num_steps):
                self._g_time.fill_(1.0 + step * dt)
                self._graph.replay()
                nxt = _advance(self._g_vel, step, self._g_current)
                self._g_current.copy_(nxt)
                traj.append(self._g_current.clone())
            current = self._g_current.clone()
        else:
            if use_graphs and self._graph is None:
                self._g_current = current
                self._g_time = torch.ones(batch, dtype=torch.float32, device=device)
                self._g_cache_k = cache_k
                self._g_cache_v = cache_v
                self._g_expert_k = expert_k
                self._g_expert_v = expert_v
                try:
                    self._capture_denoise(current, self._g_time)
                except Exception:
                    self._graph = None
                    self._graphs_ok = False
            for step in range(num_steps):
                if use_graphs and self._graph is not None:
                    if step == 0:
                        self._g_current.copy_(current)
                    self._g_time.fill_(1.0 + step * dt)
                    self._graph.replay()
                    velocity = self._g_vel
                    current = _advance(velocity, step, self._g_current)
                    self._g_current.copy_(current)
                else:
                    tau = torch.full(
                        (batch,), 1.0 + step * dt, dtype=torch.float32, device=device
                    )
                    velocity = self._denoise_step(
                        current, tau, cache_k, cache_v, expert_k, expert_v
                    )
                    current = _advance(velocity, step, current)
                traj.append(current.clone())
        if real_batch < batch:
            current = current[:real_batch]
            traj = [item[:real_batch] for item in traj]
            logps = [item[:real_batch] for item in logps]
        return (
            current,
            torch.stack(traj, dim=0),
            torch.stack(logps, dim=0),
            torch.tensor(mask, device=device, dtype=torch.bool),
            dt,
        )


@dataclass
class PreparedTask:
    text: str
    tokens: Tensor
    mask: Tensor


class SmolVLAEngine:
    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy: FlowPolicy | None = None
        self.tokenizer = None
        self.state_mean: Tensor | None = None
        self.state_std: Tensor | None = None
        self.action_mean: Tensor | None = None
        self.action_std: Tensor | None = None
        self.task: PreparedTask | None = None
        self._compute_dtype = _inference_dtype(self.device)
        self._state_buffer: Tensor | None = None
        self.source = MODEL_ID
        self.cameras = list(DEFAULT_CAMERAS)
        self.empty_cameras = 0

    def _apply_cameras(self, cameras, on_log=None) -> None:
        next_cams = _normalize_camera_names(cameras) or list(self.cameras) or list(DEFAULT_CAMERAS)
        changed = next_cams != list(self.cameras)
        self.cameras = next_cams
        if self.policy is not None:
            if int(self.policy.empty_cameras) != int(self.empty_cameras):
                changed = True
            self.policy.empty_cameras = self.empty_cameras
            if changed:
                self.policy.reset_runtime()
        if on_log is not None:
            extra = f", empty {self.empty_cameras}" if self.empty_cameras else ""
            on_log(f"Policy cameras {', '.join(self.cameras)}{extra}.")

    def load(
        self,
        source: str | None = None,
        on_log=None,
        warmup: bool = True,
        cameras=None,
    ) -> None:
        def log(text: str) -> None:
            if on_log is not None:
                on_log(text)

        source = stored_source(str(source or MODEL_ID))
        meta_empty, meta_cams = _policy_meta(source)
        chosen = _normalize_camera_names(cameras)
        self.empty_cameras = meta_empty
        if self.policy is not None and self.source == source:
            self._apply_cameras(chosen or meta_cams, on_log=log)
            log(f"Model already on {self.device}")
            return
        if self.policy is not None:
            self.policy = None
            self.task = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.source = source
        self.cameras = chosen or meta_cams or list(DEFAULT_CAMERAS)
        started = time.perf_counter()
        log(f"Loading {source} on {self.device}…")
        _configure_backends(self.device)
        weights = _load_weight_file(source)
        stats = _load_norm_stats(source)
        action_mean = _tensor_from_stats(
            stats,
            (
                f"{STATS_KEY}.buffer.action.mean",
                "action.mean",
            ),
        )
        action_std = _tensor_from_stats(
            stats,
            (
                f"{STATS_KEY}.buffer.action.std",
                "action.std",
            ),
        )
        if action_mean is None or action_std is None:
            raise RuntimeError(f"no action normalization stats in {source}")
        state_mean = _tensor_from_stats(
            stats,
            (
                f"{STATS_KEY}.buffer.observation.state.mean",
                "observation.state.mean",
                f"{STATS_KEY}.buffer.action.mean",
            ),
        )
        state_std = _tensor_from_stats(
            stats,
            (
                f"{STATS_KEY}.buffer.observation.state.std",
                "observation.state.std",
                f"{STATS_KEY}.buffer.action.std",
            ),
        )
        if state_mean is None:
            state_mean = action_mean
        if state_std is None:
            state_std = action_std
        self.state_mean = state_mean.to(self.device)
        self.state_std = state_std.to(self.device).clamp_min(NORM_EPS)
        self.action_mean = action_mean.to(self.device)
        self.action_std = action_std.to(self.device).clamp_min(NORM_EPS)
        log(f"Weights loaded ({time.perf_counter() - started:.1f}s)")
        log("Loading tokenizer…")
        self.tokenizer = AutoTokenizer.from_pretrained(VLM_ID)
        policy = FlowPolicy(self.device)
        vlm = {}
        expert = {}
        rest = {}
        for key, value in weights.items():
            if key.startswith("model.vlm_with_expert.vlm."):
                vlm[key[len("model.vlm_with_expert.vlm.") :]] = value
            elif key.startswith("model.vlm_with_expert.lm_expert."):
                expert[key[len("model.vlm_with_expert.lm_expert.") :]] = value
            elif key.startswith("vlm_with_expert.vlm."):
                vlm[key[len("vlm_with_expert.vlm.") :]] = value
            elif key.startswith("vlm_with_expert.lm_expert."):
                expert[key[len("vlm_with_expert.lm_expert.") :]] = value
            elif key.startswith("model."):
                rest[key[len("model.") :]] = value
            elif not key.startswith(("vlm_with_expert.", "optimizer", "scheduler")):
                rest[key] = value
        policy.vlm_with_expert.vlm.load_state_dict(vlm, strict=False)
        policy.vlm_with_expert.lm_expert.load_state_dict(expert, strict=True)
        policy.load_state_dict(rest, strict=False)
        policy.vlm_with_expert.fuse_qkv()
        policy.eval()
        policy.requires_grad_(False)
        if self.device.type == "cuda":
            policy.to(dtype=self._compute_dtype)
            for module in (
                policy.action_in_proj,
                policy.action_out_proj,
                policy.action_time_mlp_in,
                policy.action_time_mlp_out,
            ):
                module.to(dtype=torch.float32)
            policy.vlm_with_expert.rope_inv_freq.data = (
                policy.vlm_with_expert.rope_inv_freq.float()
            )
            policy.time_scale.data = policy.time_scale.float()
            policy.vlm_with_expert._vlm.vision_model.to(
                memory_format=torch.channels_last
            )
        self._state_buffer = torch.zeros(
            1, MAX_STATE_DIM, device=self.device, dtype=self._compute_dtype
        )
        policy.empty_cameras = self.empty_cameras
        self.policy = policy
        extra = f", empty {self.empty_cameras}" if self.empty_cameras else ""
        log(f"Policy cameras {', '.join(self.cameras)}{extra}.")
        if warmup:
            log("Warmup…")
            self._warmup()
        log("Ready")

    def _warmup(self) -> None:
        if self.policy is None or self.device.type != "cuda":
            return
        n_real = max(1, len(self.cameras))
        dummy = torch.zeros(
            n_real,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=self.device,
            dtype=self._compute_dtype,
        ).to(memory_format=torch.channels_last)
        tokens = torch.ones(1, TOKENIZER_MAX_LENGTH, dtype=torch.long, device=self.device)
        lang_mask = torch.ones(1, TOKENIZER_MAX_LENGTH, dtype=torch.bool, device=self.device)
        state = torch.zeros(1, MAX_STATE_DIM, device=self.device, dtype=self._compute_dtype)
        with torch.inference_mode():
            self.policy.sample_actions(
                dummy, n_real, tokens, lang_mask, state, num_steps=NUM_DENOISE_STEPS
            )
            self.policy.sample_actions(
                dummy, n_real, tokens, lang_mask, state, num_steps=NUM_DENOISE_STEPS
            )
            torch.cuda.synchronize()

    def prepare_task(self, instruction: str) -> PreparedTask:
        text = instruction if instruction.endswith("\n") else instruction + "\n"
        if self.task is not None and self.task.text == text:
            return self.task
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
        )
        self.task = PreparedTask(
            text=text,
            tokens=encoded["input_ids"].to(self.device),
            mask=encoded["attention_mask"].to(device=self.device, dtype=torch.bool),
        )
        if self.policy is not None:
            self.policy._lang_cache = None
        return self.task

    def _images(self, frames: dict[str, Image.Image]) -> tuple[Tensor, int]:
        names = [name for name in self.cameras if name in frames]
        cpu = [torch.from_numpy(np.asarray(frames[name].convert("RGB"))) for name in names]
        if not cpu:
            stacked = torch.zeros(
                1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device, dtype=self._compute_dtype
            )
            return stacked, 0
        if all(item.shape == cpu[0].shape for item in cpu):
            stacked = torch.stack(cpu, dim=0).to(device=self.device, non_blocking=True)
            stacked = stacked.permute(0, 3, 1, 2).to(dtype=torch.float32).mul_(1.0 / 255.0)
            stacked = _resize_pad(stacked, IMAGE_SIZE, IMAGE_SIZE).mul_(2.0).sub_(1.0)
        else:
            resized = []
            for item in cpu:
                tensor = item.to(device=self.device, non_blocking=True)
                tensor = tensor.permute(2, 0, 1)[None].to(dtype=torch.float32).mul_(1.0 / 255.0)
                resized.append(_resize_pad(tensor, IMAGE_SIZE, IMAGE_SIZE).mul_(2.0).sub_(1.0))
            stacked = torch.cat(resized, dim=0)
        stacked = stacked.to(dtype=self._compute_dtype, memory_format=torch.channels_last)
        return stacked, len(names)

    def prepare_obs(
        self,
        frames: dict[str, Image.Image],
        state: np.ndarray,
        instruction: str,
    ) -> dict:
        if self.policy is None:
            raise RuntimeError("engine is not loaded")
        task = self.prepare_task(instruction)
        images, n_real = self._images(frames)
        if n_real == 0:
            raise RuntimeError("inference requires at least one camera frame")
        state_tensor = torch.zeros(
            1, MAX_STATE_DIM, device=self.device, dtype=self._compute_dtype
        )
        joints = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state_tensor[0, :ACTION_DIM] = ((joints - self.state_mean) / self.state_std).to(
            dtype=self._compute_dtype
        )
        return {
            "images": images,
            "n_real": n_real,
            "tokens": task.tokens,
            "mask": task.mask,
            "state": state_tensor,
        }

    def prepare_obs_batch(
        self,
        items: list[tuple[dict[str, Image.Image], np.ndarray, str]],
    ) -> dict:
        if not items:
            raise RuntimeError("eval batch is empty")
        raws = [
            self.prepare_obs(frames, state, instruction)
            for frames, state, instruction in items
        ]
        return {
            "images": torch.stack([item["images"] for item in raws], dim=0),
            "n_real": int(raws[0]["n_real"]),
            "tokens": torch.cat([item["tokens"] for item in raws], dim=0),
            "mask": torch.cat([item["mask"] for item in raws], dim=0),
            "state": torch.cat([item["state"] for item in raws], dim=0),
        }

    def actions_to_env(self, actions: Tensor, index: int = 0) -> np.ndarray:
        joints = actions[index, :, :ACTION_DIM] * self.action_std + self.action_mean
        joints = torch.nan_to_num(joints.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return joints.detach().cpu().numpy()

    @torch.inference_mode()
    def predict_chunk(
        self,
        frames: dict[str, Image.Image],
        state: np.ndarray,
        instruction: str,
        num_steps: int = NUM_DENOISE_STEPS,
    ) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("engine is not loaded")
        started = time.perf_counter()
        event = {
            "timestamp": time.time(),
            "source": self.source,
            "device": str(self.device),
            "instruction": instruction,
            "cameras": list(self.cameras),
            "num_steps": int(num_steps),
        }
        try:
            obs = self.prepare_obs(frames, state, instruction)
            actions = self.policy.sample_actions(
                obs["images"],
                obs["n_real"],
                obs["tokens"],
                obs["mask"],
                obs["state"],
                num_steps=int(num_steps),
            )
            event["ok"] = True
            return self.actions_to_env(actions)
        except Exception as error:
            event["ok"] = False
            event["error"] = str(error)
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LLM_LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    @torch.inference_mode()
    def predict_chunks(
        self,
        items: list[tuple[dict[str, Image.Image], np.ndarray, str]],
        num_steps: int = NUM_DENOISE_STEPS,
    ) -> list[np.ndarray]:
        if self.policy is None:
            raise RuntimeError("engine is not loaded")
        started = time.perf_counter()
        event = {
            "timestamp": time.time(),
            "source": "eval",
            "checkpoint": str(self.source),
            "device": str(self.device),
            "batch": len(items),
            "cameras": list(self.cameras),
            "num_steps": int(num_steps),
        }
        try:
            obs = self.prepare_obs_batch(items)
            actions = self.policy.sample_actions(
                obs["images"],
                obs["n_real"],
                obs["tokens"],
                obs["mask"],
                obs["state"],
                num_steps=int(num_steps),
            )
            event["ok"] = True
            return [self.actions_to_env(actions, index) for index in range(len(items))]
        except Exception as error:
            event["ok"] = False
            event["error"] = str(error)
            raise
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LLM_LOG_PATH.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(event, ensure_ascii=False) + "\n")


_ENGINE: SmolVLAEngine | None = None


def get_engine() -> SmolVLAEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SmolVLAEngine()
    return _ENGINE


def run_smolvla_test(
    scene_dir: Path,
    config: dict,
    duration_seconds: float,
    n_action_steps: int,
    num_steps: int,
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

    engine = get_engine()
    if not cameras:
        from core.config import policy_cameras_from_config

        cameras = policy_cameras_from_config(config)
    engine.load(source=checkpoint or MODEL_ID, on_log=log, cameras=cameras)
    rollout = dict(config)
    rollout["environment"] = dict(config["environment"])
    rollout["environment"]["rollout"] = dict(config["environment"]["rollout"])
    rollout["environment"]["rollout"]["duration_seconds"] = float(duration_seconds)
    env = RandomSceneEnv(scene_dir, rollout, sensor_seed=int(config["seed"]))
    try:
        log("Reset scene…")
        observation = env.reset()
        instruction = str(env.scene_metadata["instruction"])
        engine.prepare_task(instruction)
        frames = {name: env.render_camera(name) for name in env.camera_names}
        on_frames(frames)
        steps = 0
        max_steps = max(1, int(float(duration_seconds) * env.control_hz))
        take = max(1, min(int(n_action_steps), CHUNK_SIZE))
        control_dt = 1.0 / env.control_hz
        view_dt = 1.0 / max(1.0, float(view_fps))
        last_view = 0.0
        log(
            f"Rollout {max_steps} steps at {env.control_hz:g} Hz, "
            f"chunk {take}, denoise {int(num_steps)}, "
            f"cameras {', '.join(engine.cameras)}."
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
            log(
                f"Infer #{chunk_index} at step {steps}/{max_steps} "
                f"t={steps / env.control_hz:.2f}s…"
            )
            infer_started = time.perf_counter()
            chunk = engine.predict_chunk(
                frames,
                np.asarray(observation["state"], dtype=np.float32),
                instruction,
                num_steps=int(num_steps),
            )
            infer_ms = (time.perf_counter() - infer_started) * 1000.0
            log(f"Infer #{chunk_index} done {infer_ms:.0f} ms, apply {take} actions.")
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
                    log(
                        f"Success at step {steps}/{max_steps} "
                        f"t={steps / env.control_hz:.2f}s."
                    )
                    return {
                        "success": bool(reward.success),
                        "steps": steps,
                        "seconds": steps / env.control_hz,
                        "qpos": env.data.qpos.copy(),
                        "qvel": env.data.qvel.copy(),
                    }
        if should_stop():
            log(f"Stopped at step {steps}/{max_steps}.")
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
