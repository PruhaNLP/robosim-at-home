---
license: other
task_categories:
- robotics
tags:
- mujoco
- so100
- assets
pretty_name: RoboSim at Home assets (full)
---

# RoboSim at Home — full assets

Complete simulator pack: SO-ARM100, 20 table textures, 24 indoor rooms, 50 pick targets, and 980 scanned objects.

```bash
hf download PruhaNLP/robosim-at-home-assets-full --repo-type dataset --local-dir assets
```

Smaller demo pack: [PruhaNLP/robosim-at-home-assets-demo](https://huggingface.co/datasets/PruhaNLP/robosim-at-home-assets-demo)

See `assets.yaml` for the file list.

## Sources

- Rooms and table textures: [Poly Haven](https://polyhaven.com) HDRIs/textures, [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
- Objects: [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research), CC-BY-4.0
- Robot: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `trs_so_arm100`, Apache-2.0
