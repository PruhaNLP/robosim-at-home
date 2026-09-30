"""Build PruhaNLP/TurboVLA-base from the official LIBERO TurboVLA checkpoint and push it to the Hub.

python -m model.build_turbovla_base [--push]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file
from transformers import AutoTokenizer, BertConfig, DINOv3ViTConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.turbovla import (
    ACTION_DIM,
    BASE_REPO,
    CAMERA_ORDER,
    CHUNK_SIZE,
    CONFIG_NAME,
    IMAGE_SIZE,
    STATE_DIM,
    TEXT_LENGTH,
    WEIGHTS_NAME,
    TurboVLA,
    file_to_module_keys,
    module_to_file_keys,
)

# ======Settings=========
SOURCE_REPO = "H-EmbodVis/TurboVLA"
SOURCE_CKPT = "checkpoints/libero/turbovla_libero.pth"
SOURCE_LICENSES = ("DINOv3_LICENSE.md", "LICENSE")
TOKENIZER_REPO = "google-bert/bert-base-uncased"
TARGET_REPO = BASE_REPO
PROJECT_URL = "https://github.com/PruhaNLP/robosim-at-home"
OUTPUT_DIR = REPO_ROOT / "data" / "turbovla-base"
SEED = 0
DINOV3_VITB = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
}
REINIT_MODULES = (
    "action_head.state_projection.net.0.",
    "action_head.state_projection.net.1.",
    "action_head.decoder.action_projection.layers.2.",
)
INTERACTION_KEYS = (
    "hidden_dim",
    "nheads",
    "num_layers",
    "dim_feedforward",
    "enhancer_inner_dim",
    "text_dropout",
    "fusion_dropout",
    "fusion_droppath",
)
ACTION_KEYS = ("num_state_tokens", "num_layers", "mlp_hidden_dim", "state_hidden_dim", "dropout")
# ======Settings=========

MODEL_CARD = """---
license: other
license_name: dinov3-license
license_link: https://huggingface.co/{target_repo}/blob/main/DINOv3_LICENSE.md
base_model: {source_repo}
pipeline_tag: robotics
library_name: pytorch
tags:
- robotics
- vision-language-action
- turbovla
- so100
- lerobot
- dinov3
- bert
arxiv: "2607.27205"
---

# TurboVLA-base (SO-100)

A ready-to-fine-tune [TurboVLA](https://arxiv.org/abs/2607.27205) checkpoint for the SO-100 / SO-101 arm, in the spirit of `lerobot/smolvla_base`.
No manual model assembly: one folder holds the config, the full weights (DINOv3 ViT-B + BERT + interaction + action decoder) and the tokenizer.

TurboVLA drops the LLM from the VLA loop. DINOv3 encodes each camera, BERT encodes the instruction, {layers} bidirectional
vision-language cross-attention layers fuse them, and an ACT-style decoder predicts a {chunk}-step continuous action chunk in one pass
(~0.2B params, L1 behavior cloning).

## How it was made

Built by [`model/build_turbovla_base.py`]({project}/blob/main/model/build_turbovla_base.py) from the official LIBERO checkpoint
[`{source_repo}/{source_ckpt}`](https://huggingface.co/{source_repo}), the same starting point the paper uses for real-robot fine-tuning.
Every tensor is copied as is, except the embodiment-specific ones, which are freshly initialized (seed {seed}):

{reinit}

## I/O contract

| | |
| --- | --- |
| Cameras | {n_cameras} RGB views, order `{cameras}`, resized + padded to {image_size}x{image_size}, ImageNet mean/std |
| Instruction | English text, BERT uncased, padded to {text_length} tokens |
| State | {state_dim}-D SO-100 joint positions, normalized with mean/std |
| Action | {chunk} x {action_dim} absolute joint targets, `tanh` output mapped from per-joint dataset min/max |

This base has **no normalization stats**: they come from your dataset on the first fine-tune and are saved as `stats.safetensors`.
The paper recipe freezes BERT and trains everything else with lr 5e-5, AdamW (0.9, 0.95), L1 loss.

## Use it

In [robosim-at-home]({project}): switch **Mode → TurboVLA**, open **Train**, pick `{target_repo}`, tick datasets, start.
Inference and Eval then list the fine-tuned runs.

Standalone, with [`model/turbovla.py`]({project}/blob/main/model/turbovla.py):

```python
from model.turbovla import TurboEngine

engine = TurboEngine(device="cuda")
engine.load("{target_repo}")          # config + weights + tokenizer from this repo
# after fine-tuning (stats present):
chunk = engine.predict_chunk({{"front": front_pil, "wrist": wrist_pil}}, joints, "pick up the cube and put it in the tray")
```

## Files

| File | Content |
| --- | --- |
| `config.json` | `type: turbovla`, cameras, sizes, interaction / action head, full DINOv3 and BERT configs |
| `model.safetensors` | fp32 weights; key names match the official `turbovla.models.turbovla.TurboVLA` module (transformers 4.x DINOv3 layout `vision_encoder.backbone.layer.N`) |
| `tokenizer.json`, `tokenizer_config.json` | `{tokenizer_repo}` tokenizer |
| `DINOv3_LICENSE.md`, `LICENSE` | licenses inherited from `{source_repo}` |

The weights load strictly into the official TurboVLA class built with `action_dim={action_dim}`, `state_dim={state_dim}`, `num_views={n_cameras}`,
`image_size={image_size}`, `text.padding_length={text_length}`, and give the same outputs as `model/turbovla.py`.

## License

Weights contain DINOv3-derived parameters and are distributed under the [DINOv3 License](DINOv3_LICENSE.md); TurboVLA code is Apache-2.0.

## Citation

```bibtex
@article{{xie2026turbovla,
  title   = {{TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM}},
  author  = {{Xie, Hengyi and Yao, Chenfei and Wu, Xianjin and Xi, Xuanyang and Tang, Yiping and Xu, Di and
             Zhu, Yingying and Liang, Dingkang and Bai, Xiang and Ding, Han}},
  journal = {{arXiv preprint arXiv:2607.27205}},
  year    = {{2026}}
}}
```
"""


def build_config(official: dict) -> dict:
    return {
        "type": "turbovla",
        "paper": "https://arxiv.org/abs/2607.27205",
        "init_from": f"{SOURCE_REPO}/{SOURCE_CKPT}",
        "cameras": list(CAMERA_ORDER),
        "image_size": IMAGE_SIZE,
        "chunk_size": CHUNK_SIZE,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "text_length": TEXT_LENGTH,
        "vision": {"dropout": float(official["vision"]["dropout"])},
        "interaction": {key: official["interaction"][key] for key in INTERACTION_KEYS},
        "action": {key: official["action"][key] for key in ACTION_KEYS},
        "dinov3": DINOv3ViTConfig(**DINOV3_VITB).to_dict(),
        "bert": BertConfig.from_pretrained(TOKENIZER_REPO).to_dict(),
    }


def transplant(model: TurboVLA, source: dict[str, torch.Tensor]) -> list[str]:
    target = model.state_dict()
    unused = sorted(set(source) - set(target))
    if unused:
        raise RuntimeError(f"source tensors without a target: {unused[:8]}")
    reinit = []
    for key, value in target.items():
        if key.startswith(REINIT_MODULES):
            reinit.append(key)
            continue
        src = source.get(key)
        if src is None or tuple(src.shape) != tuple(value.shape):
            raise RuntimeError(f"cannot copy {key}: source {None if src is None else tuple(src.shape)}")
        target[key] = src.to(value.dtype)
    model.load_state_dict(target, strict=True)
    return reinit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--push", action="store_true", help=f"upload to {TARGET_REPO}")
    args = parser.parse_args()

    ckpt_path = hf_hub_download(SOURCE_REPO, SOURCE_CKPT)
    print(f"Source {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    official = checkpoint["model_config"]
    source = file_to_module_keys(checkpoint["model_state_dict"])
    config = build_config(official)

    torch.manual_seed(SEED)
    model = TurboVLA(config)
    reinit = transplant(model, source)
    target = model.state_dict()
    total = sum(value.numel() for value in target.values())
    print(f"Copied {len(target) - len(reinit)} tensors, reinit {len(reinit)}: {reinit}")
    print(f"Parameters {total / 1e6:.1f}M")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    state = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
    save_file(module_to_file_keys(state), str(OUTPUT_DIR / WEIGHTS_NAME), metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(str(OUTPUT_DIR))
    for name in SOURCE_LICENSES:
        shutil.copy2(hf_hub_download(SOURCE_REPO, name), OUTPUT_DIR / name)
    (OUTPUT_DIR / "README.md").write_text(
        MODEL_CARD.format(
            project=PROJECT_URL,
            source_repo=SOURCE_REPO,
            source_ckpt=SOURCE_CKPT,
            target_repo=TARGET_REPO,
            tokenizer_repo=TOKENIZER_REPO,
            seed=SEED,
            reinit="\n".join(f"- `{key}`" for key in reinit),
            layers=config["interaction"]["num_layers"],
            chunk=CHUNK_SIZE,
            n_cameras=len(CAMERA_ORDER),
            cameras=", ".join(CAMERA_ORDER),
            image_size=IMAGE_SIZE,
            text_length=TEXT_LENGTH,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
        )
    )
    print(f"Wrote {OUTPUT_DIR}")

    if args.push:
        api = HfApi()
        api.create_repo(TARGET_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=TARGET_REPO,
            folder_path=str(OUTPUT_DIR),
            commit_message=f"TurboVLA-base for SO-100 from {SOURCE_REPO}/{SOURCE_CKPT}",
        )
        print(f"Pushed https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    main()
