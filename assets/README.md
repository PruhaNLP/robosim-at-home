---
license: other
task_categories:
- robotics
tags:
- mujoco
- so100
- assets
pretty_name: RoboSim at Home assets (demo)
---

# RoboSim at Home — demo assets

The ACT training scene: robot, 6 objects, 1 HDRI, 1 table. Target vs distractor is chosen in the studio, not in this pack.

```bash
hf download PruhaNLP/robosim-at-home-assets-demo robosim-at-home-assets-demo.tar.gz --repo-type dataset
tar -xzf robosim-at-home-assets-demo.tar.gz
```

Full pack: [PruhaNLP/robosim-at-home-assets-full](https://huggingface.co/datasets/PruhaNLP/robosim-at-home-assets-full)

See `assets.yaml` for the file list.

## Sources

- Rooms and table textures: [Poly Haven](https://polyhaven.com) HDRIs/textures, [CC0](https://creativecommons.org/publicdomain/zero/1.0/)
- Objects: [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research), CC-BY-4.0
- Robot: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `trs_so_arm100`, Apache-2.0
