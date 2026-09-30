"""TurboVLA (arXiv 2607.27205): DINOv3 + BERT, bidirectional V-L fusion, ACT chunk decoder."""
from __future__ import annotations

import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import AutoTokenizer, BertConfig, BertModel, DINOv3ViTConfig, DINOv3ViTModel

_SIM_ROOT = Path(__file__).resolve().parents[1] / "sim"
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

from core.config import LLM_LOG_PATH, resolve_source
from model.act import _resize_pad

# ======Settings=========
BASE_REPO = "PruhaNLP/TurboVLA-base"
IMAGE_SIZE = 256
CHUNK_SIZE = 12
ACTION_DIM = 6
STATE_DIM = 6
CAMERA_ORDER = ("front", "wrist")
TEXT_LENGTH = 32
BERT_SPECIAL_TOKENS = ("[CLS]", "[SEP]", ".", "?")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORM_EPS = 1e-6
TEXT_CACHE_LIMIT = 512
ATTN_CLAMP = 50000.0
CONFIG_NAME = "config.json"
STATS_NAME = "stats.safetensors"
WEIGHTS_NAME = "model.safetensors"
DINOV3_FILE_PREFIX = "vision_encoder.backbone.layer."
DINOV3_MODULE_PREFIX = "vision_encoder.backbone.model.layer."
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


def is_turbovla_checkpoint(source: str) -> bool:
    raw = str(source or "").strip()
    if raw == BASE_REPO:
        return True
    path = _existing_dir(raw)
    if path is None:
        return False
    cfg = _unwrap_pretrained(path) / CONFIG_NAME
    if cfg.is_file():
        return str(json.loads(cfg.read_text()).get("type") or "") == "turbovla"
    return False


def _checkpoint_dir(source: str, log) -> Path:
    path = _existing_dir(source)
    if path is None:
        raw = str(source or "").strip() or BASE_REPO
        log(f"Download TurboVLA {raw} from Hugging Face.")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=raw))
    return _unwrap_pretrained(path)


def file_to_module_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        (DINOV3_MODULE_PREFIX + key[len(DINOV3_FILE_PREFIX):] if key.startswith(DINOV3_FILE_PREFIX) else key): value
        for key, value in state.items()
    }


def module_to_file_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        (DINOV3_FILE_PREFIX + key[len(DINOV3_MODULE_PREFIX):] if key.startswith(DINOV3_MODULE_PREFIX) else key): value
        for key, value in state.items()
    }


def _amp_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    from train_loop.amp import resolve_amp

    enabled, dtype, _ = resolve_amp(device, True)
    return dtype if enabled else None


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask.div_(keep)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + dims, dims + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < last else layer(x)
        return x


class BiMultiHeadAttention(nn.Module):
    def __init__(self, dim: int, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.v_proj = nn.Linear(dim, embed_dim)
        self.l_proj = nn.Linear(dim, embed_dim)
        self.values_v_proj = nn.Linear(dim, embed_dim)
        self.values_l_proj = nn.Linear(dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, dim)
        self.out_l_proj = nn.Linear(embed_dim, dim)
        for layer in (
            self.v_proj,
            self.l_proj,
            self.values_v_proj,
            self.values_l_proj,
            self.out_v_proj,
            self.out_l_proj,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return (
            x.view(batch, length, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .reshape(batch * self.num_heads, length, self.head_dim)
        )

    def _merge(self, x: Tensor, batch: int) -> Tensor:
        length = x.shape[1]
        return (
            x.view(batch, self.num_heads, length, self.head_dim)
            .transpose(1, 2)
            .reshape(batch, length, self.embed_dim)
        )

    def forward(self, v: Tensor, l: Tensor, pad_l: Tensor | None) -> tuple[Tensor, Tensor]:
        batch = v.shape[0]
        query = self._heads(self.v_proj(v) * self.scale)
        key = self._heads(self.l_proj(l))
        value_v = self._heads(self.values_v_proj(v))
        value_l = self._heads(self.values_l_proj(l))
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = (attn - attn.max()).clamp(-ATTN_CLAMP, ATTN_CLAMP)
        attn_l = attn.transpose(1, 2)
        attn_l = (attn_l - attn_l.max(dim=-1, keepdim=True)[0]).clamp(-ATTN_CLAMP, ATTN_CLAMP)
        attn_l = attn_l.softmax(dim=-1)
        if pad_l is not None:
            mask = pad_l[:, None, None, :].expand(-1, self.num_heads, 1, -1).flatten(0, 1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn_v = attn.softmax(dim=-1)
        out_v = torch.bmm(F.dropout(attn_v, p=self.dropout, training=self.training), value_l)
        out_l = torch.bmm(F.dropout(attn_l, p=self.dropout, training=self.training), value_v)
        return self.out_v_proj(self._merge(out_v, batch)), self.out_l_proj(self._merge(out_l, batch))


class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        drop_path: float,
        init_values: float = 1e-4,
    ) -> None:
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(dim)
        self.layer_norm_l = nn.LayerNorm(dim)
        self.attn = BiMultiHeadAttention(dim, embed_dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_l = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, v: Tensor, l: Tensor, pad_l: Tensor | None) -> tuple[Tensor, Tensor]:
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, pad_l)
        return v + self.drop_path(self.gamma_v * delta_v), l + self.drop_path(self.gamma_l * delta_l)


class TextLayer(nn.Module):
    def __init__(self, dim: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, blocked: Tensor) -> Tensor:
        src = x.transpose(0, 1)
        # Head-major repeat (not batch-major) is what the Grounding DINO / TurboVLA weights were trained with.
        mask = blocked.repeat(self.nhead, 1, 1)
        out = self.self_attn(src, src, src, attn_mask=mask, need_weights=False)[0]
        src = self.norm1(src + self.dropout1(out))
        src = self.norm2(src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src))))))
        return src.transpose(0, 1)


class VisionLanguageInteraction(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        heads = max(1, int(cfg["nheads"]) // 2)
        inner = int(cfg["enhancer_inner_dim"])
        layers = int(cfg["num_layers"])
        self.text_layers = nn.ModuleList(
            TextLayer(hidden, heads, inner, float(cfg["text_dropout"])) for _ in range(layers)
        )
        self.fusion_layers = nn.ModuleList(
            BiAttentionBlock(hidden, inner, heads, float(cfg["fusion_dropout"]), float(cfg["fusion_droppath"]))
            for _ in range(layers)
        )

    def forward(self, v: Tensor, l: Tensor, pad_l: Tensor, allowed_l: Tensor) -> tuple[Tensor, Tensor]:
        blocked = ~allowed_l
        for fusion, text in zip(self.fusion_layers, self.text_layers):
            v, l = fusion(v, l, pad_l)
            l = text(l, blocked)
        return v, l


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class VisionEncoder(nn.Module):
    def __init__(self, dinov3: dict) -> None:
        super().__init__()
        self.backbone = DINOv3ViTModel(DINOv3ViTConfig.from_dict(dinov3))
        self.backbone.embeddings.mask_token.requires_grad_(False)
        cfg = self.backbone.config
        self.hidden_size = int(cfg.hidden_size)
        self.prefix_tokens = int(cfg.num_register_tokens) + 1
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True
        self.backbone.requires_grad_(False)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch, views = pixel_values.shape[:2]
        with torch.no_grad() if self.frozen else nullcontext():
            tokens = self.backbone(pixel_values=pixel_values.flatten(0, 1)).last_hidden_state[:, self.prefix_tokens:]
        return tokens.reshape(batch, views, tokens.shape[1], tokens.shape[2])


class TextEncoder(nn.Module):
    def __init__(self, bert: dict, hidden_dim: int) -> None:
        super().__init__()
        self.bert = BertModel(BertConfig.from_dict(bert))
        self.bert.requires_grad_(False)
        self.bert.eval()
        self.text_projection = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    @torch.no_grad()
    def encode(self, input_ids: Tensor, positions: Tensor, allowed: Tensor) -> Tensor:
        additive = (1.0 - allowed[:, None].float()) * torch.finfo(torch.float32).min
        hidden = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=positions,
        )
        return self.bert.encoder(hidden, attention_mask=additive)[0]


class StateProjection(nn.Module):
    def __init__(self, cfg: dict, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(cfg["num_state_tokens"])
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, int(cfg["state_hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(int(cfg["state_hidden_dim"]), self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: Tensor) -> Tensor:
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position)


class ACTDecoder(nn.Module):
    def __init__(self, cfg: dict, chunk_size: int, action_dim: int, hidden_dim: int, nheads: int, ff: int) -> None:
        super().__init__()
        self.action_queries = nn.Embedding(chunk_size, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=ff,
                dropout=float(cfg["dropout"]),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(cfg["num_layers"]),
        )
        self.action_projection = MLP(hidden_dim, int(cfg["mlp_hidden_dim"]), action_dim, 3)

    def forward(self, memory: Tensor) -> Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        return torch.tanh(self.action_projection(self.decoder(tgt=queries, memory=memory)))


class ActionHead(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        inter = config["interaction"]
        hidden = int(inter["hidden_dim"])
        self.state_projection = StateProjection(config["action"], int(config["state_dim"]), hidden)
        self.decoder = ACTDecoder(
            config["action"],
            int(config["chunk_size"]),
            int(config["action_dim"]),
            hidden,
            int(inter["nheads"]),
            int(inter["dim_feedforward"]),
        )

    def forward(self, tokens: Tensor, state: Tensor) -> Tensor:
        return self.decoder(torch.cat([tokens, self.state_projection(state)], dim=1))


class TurboVLA(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = int(config["interaction"]["hidden_dim"])
        views = len(config["cameras"])
        self.text_encoder = TextEncoder(config["bert"], hidden)
        self.vision_encoder = VisionEncoder(config["dinov3"])
        self.vision_projection = VisionProjection(
            self.vision_encoder.hidden_size,
            hidden,
            max(hidden * 4, self.vision_encoder.hidden_size // 2),
            float(config["vision"]["dropout"]),
        )
        self.view_embedding = nn.Parameter(torch.zeros(1, views, hidden))
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        self.vision_language_interaction = VisionLanguageInteraction(config["interaction"])
        self.action_head = ActionHead(config)
        self.amp_dtype: torch.dtype | None = None

    def _autocast(self, device: torch.device):
        if self.amp_dtype is None or device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def encode_vision(self, images: Tensor) -> Tensor:
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        with self._autocast(images.device):
            tokens = self.vision_encoder((images - mean) / std)
        tokens = self.vision_projection(tokens.float())
        return (tokens + self.view_embedding[:, :, None, :]).flatten(1, 2)

    def forward(self, images: Tensor, text: tuple[Tensor, Tensor, Tensor], state: Tensor) -> Tensor:
        bert_hidden, pad_l, allowed_l = text
        v = self.encode_vision(images)
        l = self.text_encoder.text_projection(bert_hidden)
        v, l = self.vision_language_interaction(v, l, pad_l, allowed_l)
        return self.action_head(torch.cat([v, l], dim=1), state)


def _sub_sentence_masks(ids: Tensor, special: Tensor) -> tuple[Tensor, Tensor]:
    length = ids.numel()
    allowed = torch.eye(length, dtype=torch.bool)
    positions = torch.zeros(length, dtype=torch.long)
    previous = 0
    for col in torch.nonzero(torch.isin(ids, special)).flatten().tolist():
        if col == 0 or col == length - 1:
            allowed[col, col] = True
            positions[col] = 0
        else:
            allowed[previous + 1 : col + 1, previous + 1 : col + 1] = True
            positions[previous + 1 : col + 1] = torch.arange(col - previous)
        previous = col
    return allowed, positions


class TurboEngine:
    def __init__(self, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy: TurboVLA | None = None
        self.config: dict | None = None
        self.tokenizer = None
        self.special_ids: Tensor | None = None
        self.cameras: list[str] = list(CAMERA_ORDER)
        self.state_mean: Tensor | None = None
        self.state_std: Tensor | None = None
        self.action_min: Tensor | None = None
        self.action_max: Tensor | None = None
        self.source = BASE_REPO
        self._text_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    def load(self, source: str, on_log=None) -> None:
        def log(text: str) -> None:
            if on_log is not None:
                on_log(text)

        path = _checkpoint_dir(source, log)
        cfg_path = path / CONFIG_NAME
        if not cfg_path.is_file():
            raise RuntimeError(f"no {CONFIG_NAME} in {path}")
        config = json.loads(cfg_path.read_text())
        if str(config.get("type") or "") != "turbovla":
            raise RuntimeError(f"{path} is not a TurboVLA checkpoint")
        weights = path / WEIGHTS_NAME
        if not weights.is_file():
            raise RuntimeError(f"TurboVLA checkpoint is missing {WEIGHTS_NAME} in {path}")
        log(f"Loading TurboVLA {path} on {self.device}…")
        torch.set_float32_matmul_precision("high")
        policy = TurboVLA(config)
        policy.load_state_dict(file_to_module_keys(load_file(str(weights))))
        policy.amp_dtype = _amp_dtype(self.device)
        self.policy = policy.to(self.device).eval()
        self.config = config
        self.cameras = [str(name) for name in config["cameras"]]
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.special_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(BERT_SPECIAL_TOKENS)))
        self._text_cache = {}
        stats_path = path / STATS_NAME
        if stats_path.is_file():
            stats = load_file(str(stats_path))
            self.set_stats(
                stats["observation.state.mean"],
                stats["observation.state.std"],
                stats["action.min"],
                stats["action.max"],
            )
        else:
            self.state_mean = self.state_std = self.action_min = self.action_max = None
        self.source = str(source or BASE_REPO)
        log(f"TurboVLA ready · amp {policy.amp_dtype or 'off'} · cameras {', '.join(self.cameras)}")

    def has_stats(self) -> bool:
        return self.action_min is not None

    def set_stats(self, state_mean: Tensor, state_std: Tensor, action_min: Tensor, action_max: Tensor) -> None:
        self.state_mean = state_mean.to(self.device).float().view(-1)[:STATE_DIM]
        self.state_std = state_std.to(self.device).float().view(-1)[:STATE_DIM].clamp_min(NORM_EPS)
        self.action_min = action_min.to(self.device).float().view(-1)[:ACTION_DIM]
        self.action_max = action_max.to(self.device).float().view(-1)[:ACTION_DIM]

    def set_cameras(self, cameras: list[str]) -> None:
        if self.policy is None or self.config is None:
            raise RuntimeError("TurboVLA engine is empty")
        old = self.policy.view_embedding.data
        if len(cameras) != old.shape[1]:
            fresh = torch.empty(1, len(cameras), old.shape[2], device=old.device, dtype=old.dtype)
            nn.init.trunc_normal_(fresh, std=0.02)
            keep = min(len(cameras), old.shape[1])
            fresh[:, :keep] = old[:, :keep]
            self.policy.view_embedding = nn.Parameter(fresh)
        self.cameras = list(cameras)
        self.config["cameras"] = list(cameras)

    def map_cameras(self, cameras) -> list[str]:
        chosen = [name for name in dict.fromkeys(str(item or "").strip() for item in cameras or []) if name]
        if chosen:
            slots = len(self.cameras)
            self.cameras = chosen[:slots] + [f"__empty_{index}" for index in range(len(chosen), slots)]
        return chosen

    def save(self, dest: Path) -> None:
        if self.policy is None or self.config is None or not self.has_stats():
            raise RuntimeError("TurboVLA engine is empty")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / CONFIG_NAME).write_text(json.dumps(self.config, indent=2) + "\n")
        state = {key: value.detach().cpu().contiguous() for key, value in self.policy.state_dict().items()}
        save_file(module_to_file_keys(state), str(dest / WEIGHTS_NAME))
        save_file(
            {
                "observation.state.mean": self.state_mean.detach().cpu(),
                "observation.state.std": self.state_std.detach().cpu(),
                "action.min": self.action_min.detach().cpu(),
                "action.max": self.action_max.detach().cpu(),
            },
            str(dest / STATS_NAME),
        )
        self.tokenizer.save_pretrained(str(dest))

    def encode_text(self, instructions: list[str]) -> tuple[Tensor, Tensor, Tensor]:
        if self.policy is None:
            raise RuntimeError("TurboVLA engine is not loaded")
        missing = [text for text in dict.fromkeys(instructions) if text not in self._text_cache]
        if missing:
            if len(self._text_cache) + len(missing) > TEXT_CACHE_LIMIT:
                self._text_cache = {}
            tokens = self.tokenizer(
                missing,
                padding="max_length",
                truncation=True,
                max_length=TEXT_LENGTH,
                return_tensors="pt",
            )
            masks = [_sub_sentence_masks(row, self.special_ids) for row in tokens["input_ids"]]
            allowed = torch.stack([item[0] for item in masks]).to(self.device)
            positions = torch.stack([item[1] for item in masks]).to(self.device)
            hidden = self.policy.text_encoder.encode(tokens["input_ids"].to(self.device), positions, allowed)
            pad = ~tokens["attention_mask"].bool().to(self.device)
            for index, text in enumerate(missing):
                self._text_cache[text] = (hidden[index].float(), pad[index], allowed[index])
        rows = [self._text_cache[text] for text in instructions]
        return tuple(torch.stack([row[part] for row in rows]) for part in range(3))

    def _stack_frames(self, frames: dict[str, Image.Image]) -> Tensor:
        views = []
        for name in self.cameras:
            if name not in frames:
                views.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, device=self.device))
                continue
            image = torch.from_numpy(np.array(frames[name].convert("RGB"))).to(self.device, non_blocking=True)
            image = image.permute(2, 0, 1)[None].float().div_(255.0)
            views.append(_resize_pad(image, IMAGE_SIZE)[0])
        return torch.stack(views, 0)

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean) / self.state_std

    def normalize_actions(self, actions: Tensor) -> Tensor:
        span = (self.action_max - self.action_min).clamp_min(NORM_EPS)
        return (2.0 * (actions - self.action_min) / span - 1.0).clamp(-1.0, 1.0)

    def actions_to_env(self, actions: Tensor, index: int = 0) -> np.ndarray:
        chunk = 0.5 * (actions[index].float() + 1.0) * (self.action_max - self.action_min) + self.action_min
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        return chunk.detach().cpu().numpy()

    def prepare_obs_batch(self, items: list[tuple[dict[str, Image.Image], np.ndarray, str]]) -> dict:
        if self.policy is None or not self.has_stats():
            raise RuntimeError("TurboVLA engine has no normalization stats; fine-tune it first")
        images = torch.stack([self._stack_frames(frames) for frames, _, _ in items], 0)
        states = torch.stack(
            [torch.as_tensor(state, dtype=torch.float32).view(-1)[:STATE_DIM] for _, state, _ in items], 0
        ).to(self.device)
        return {
            "images": images,
            "state": self.normalize_state(states),
            "text": self.encode_text([str(instruction or "") for _, _, instruction in items]),
        }

    def _log_call(self, event: dict, started: float) -> None:
        event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LLM_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False) + "\n")

    @torch.inference_mode()
    def predict_chunks(
        self,
        items: list[tuple[dict[str, Image.Image], np.ndarray, str]],
        num_steps: int | None = None,
        source: str = "eval",
    ) -> list[np.ndarray]:
        if self.policy is None:
            raise RuntimeError("TurboVLA engine is not loaded")
        started = time.perf_counter()
        event = {
            "timestamp": time.time(),
            "source": source,
            "model": "turbovla",
            "checkpoint": str(self.source),
            "device": str(self.device),
            "batch": len(items),
            "instruction": [str(item[2] or "") for item in items] if len(items) > 1 else str(items[0][2] or ""),
            "cameras": list(self.cameras),
        }
        try:
            self.policy.eval()
            obs = self.prepare_obs_batch(items)
            actions = self.policy(obs["images"], obs["text"], obs["state"])
            event["ok"] = True
            return [self.actions_to_env(actions, index) for index in range(len(items))]
        except Exception as error:
            event["ok"] = False
            event["error"] = str(error)
            raise
        finally:
            self._log_call(event, started)

    def predict_chunk(self, frames: dict[str, Image.Image], state: np.ndarray, instruction: str = "") -> np.ndarray:
        return self.predict_chunks([(frames, state, instruction)], source="infer")[0]


class TurboRamDataset(torch.utils.data.Dataset):
    def __init__(self, images: Tensor, states: Tensor, actions: Tensor, ends: Tensor, task_ids: Tensor) -> None:
        self.images = images
        self.states = states
        self.actions = actions
        self.ends = ends
        self.task_ids = task_ids

    def __len__(self) -> int:
        return int(self.states.size(0))

    def __getitem__(self, index: int):
        end = int(self.ends[index])
        take = min(CHUNK_SIZE, end - index)
        chunk = torch.zeros(CHUNK_SIZE, ACTION_DIM, dtype=torch.float32)
        mask = torch.zeros(CHUNK_SIZE, dtype=torch.float32)
        chunk[:take] = self.actions[index : index + take]
        mask[:take] = 1.0
        return self.images[index].float().div_(255.0), self.states[index], chunk, mask, self.task_ids[index]


def run_turbovla_sft(
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
    training: dict | None = None,
    freeze_vision: bool = False,
    vision_lr: float | None = None,
) -> None:
    from model.act import _act_lr, _load_collect_cache
    from train_loop.amp import make_scaler, resolve_amp

    training = training or {}
    engine = TurboEngine(device=str(device))
    engine.load(source or BASE_REPO, on_log=log)
    if list(cameras) != engine.cameras:
        log(f"Camera slots {', '.join(engine.cameras)} → {', '.join(cameras)}.")
        engine.set_cameras(list(cameras))
    images, states, actions, ends = _load_collect_cache(
        episodes,
        engine.cameras,
        device,
        log,
        progress=progress,
        should_stop=should_stop,
    )
    if should_stop():
        return
    tasks = [str(item.get("task") or "") for item in episodes]
    order = {int(end): index for index, end in enumerate(torch.unique(ends).tolist())}
    task_ids = torch.tensor([order[int(end)] for end in ends.tolist()], dtype=torch.long)
    if engine.has_stats():
        log("Keep normalization stats from the checkpoint.")
    else:
        engine.set_stats(
            states.mean(0),
            states.std(0).clamp_min(NORM_EPS),
            actions.min(0).values,
            actions.max(0).values,
        )
        log("Normalization stats from the selected datasets.")
    log(
        f"Frame cache {images.size(0)} frames · {images.size(1)} cameras · "
        f"{IMAGE_SIZE}px · {images.nbytes / 1024**3:.2f} GB · {len(set(tasks))} instructions."
    )
    loader = torch.utils.data.DataLoader(
        TurboRamDataset(images, states, actions, ends, task_ids),
        batch_size=max(1, int(batch)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    steps = max(1, epochs * max(1, math.ceil(len(loader.dataset) / max(1, int(batch)))))
    progress("train", 0, steps, f"Start TurboVLA · {steps} steps")
    peak_lr = float(lr)
    decay_lr = max(0.0, float(training.get("scheduler_decay_lr", 2.5e-6)))
    warmup = max(0, int(training.get("scheduler_warmup_steps", 1000)))
    decay_steps = max(0, int(training.get("scheduler_decay_steps", 30000)))
    use_amp, amp_dtype, needs_scaler = resolve_amp(device, bool(training.get("use_amp", True)))
    scaler = make_scaler(needs_scaler, device)
    policy = engine.policy
    policy.amp_dtype = amp_dtype if use_amp else None
    vision_lr = peak_lr if vision_lr is None else float(vision_lr)
    if freeze_vision:
        policy.vision_encoder.freeze()
    backbone = {id(param) for param in policy.vision_encoder.backbone.parameters()}
    params = [param for param in policy.parameters() if param.requires_grad]
    groups = [
        {"params": [p for p in params if id(p) not in backbone], "lr": peak_lr, "base_lr": peak_lr},
        {"params": [p for p in params if id(p) in backbone], "lr": vision_lr, "base_lr": vision_lr},
    ]
    optim = torch.optim.AdamW(
        [group for group in groups if group["params"]],
        lr=peak_lr,
        weight_decay=max(0.0, float(training.get("optimizer_weight_decay", 1e-10))),
        betas=(float(training.get("optimizer_beta1", 0.9)), float(training.get("optimizer_beta2", 0.95))),
        eps=max(0.0, float(training.get("optimizer_eps", 1e-8))),
    )
    grad_clip = max(0.0, float(training.get("grad_clip_norm", 10.0)))
    log(
        f"TurboVLA optim · {sum(p.numel() for p in params) / 1e6:.1f}M trainable · BERT frozen · "
        f"vision {'frozen' if freeze_vision else f'lr {vision_lr:g}'} · head lr {peak_lr:g} · "
        f"amp {amp_dtype if use_amp else 'off'} · warmup {warmup} · decay {decay_steps} → {decay_lr:g}."
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

    policy.train()
    try:
        for epoch in range(1, epochs + 1):
            if should_stop():
                break
            for images_b, state, action_b, mask, task_b in loader:
                if should_stop():
                    break
                step += 1
                lr_now = _act_lr(step, peak_lr, decay_lr, warmup, decay_steps, steps)
                for group in optim.param_groups:
                    group["lr"] = lr_now * group["base_lr"] / peak_lr
                images_b = images_b.to(device, non_blocking=True)
                state = engine.normalize_state(state.to(device, non_blocking=True))
                target = engine.normalize_actions(action_b.to(device, non_blocking=True))
                mask = mask.to(device, non_blocking=True)
                text = engine.encode_text([tasks[int(index)] for index in task_b])
                optim.zero_grad(set_to_none=True)
                pred = policy(images_b, text, state)
                per = (pred.float() - target).abs().mean(dim=-1)
                loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                else:
                    loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip > 0 else float("inf")))
                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                if step == 1 or step % 10 == 0 or step >= steps:
                    log(
                        f"step:{step} smpl:{step * images_b.size(0)} epch:{epoch} "
                        f"loss:{float(loss):.4f} grdn:{grad:.3f} lr:{lr_now:g} "
                        f"smp/s:{step * images_b.size(0) / max(time.monotonic() - started, 1e-6):.1f}"
                    )
                progress("train", step, steps, f"Step {step}/{steps}")
            if save_every > 0 and epoch % save_every == 0 and step > 0:
                dest = write_ckpt(f"{step:06d}")
                log(f"Saved {dest}.")
    finally:
        if step > 0:
            dest = write_ckpt(f"{step:06d}")
            log(f"Saved {dest}.")


def run_turbovla_test(
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

    engine = TurboEngine()
    engine.load(checkpoint or BASE_REPO, on_log=log)
    if not engine.has_stats():
        raise RuntimeError("TurboVLA base has no SO-100 stats yet; fine-tune it on the Train page first")
    if not cameras:
        from core.config import policy_cameras_from_config

        cameras = policy_cameras_from_config(config)
    slots = len(engine.cameras)
    chosen = engine.map_cameras(cameras)
    if chosen and len(chosen) != slots:
        log(f"TurboVLA checkpoint has {slots} camera slots.")
    rollout = dict(config)
    rollout["environment"] = dict(config["environment"])
    rollout["environment"]["rollout"] = dict(config["environment"]["rollout"])
    rollout["environment"]["rollout"]["duration_seconds"] = float(duration_seconds)
    env = RandomSceneEnv(scene_dir, rollout, sensor_seed=int(config["seed"]))
    try:
        log("Reset scene…")
        observation = env.reset()
        instruction = str(env.scene_metadata["instruction"])
        frames = {name: env.render_camera(name) for name in env.camera_names}
        on_frames(frames)
        steps = 0
        max_steps = max(1, int(float(duration_seconds) * env.control_hz))
        take = max(1, min(int(n_action_steps), CHUNK_SIZE))
        control_dt = 1.0 / env.control_hz
        view_dt = 1.0 / max(1.0, float(view_fps))
        last_view = 0.0
        log(
            f"TurboVLA rollout {max_steps} steps at {env.control_hz:g} Hz, "
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
            started = time.perf_counter()
            chunk = engine.predict_chunk(frames, np.asarray(observation["state"], dtype=np.float32), instruction)
            log(f"Infer #{chunk_index} at step {steps}/{max_steps} · {(time.perf_counter() - started) * 1000.0:.0f} ms.")
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
