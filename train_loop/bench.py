from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ======Settings=========
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "sim") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "sim"))
from core.config import BENCH_DIR, LEROBOT_DIR, ensure_data_dirs, stored_path

DATASET_ROOT = LEROBOT_DIR / "smolvla"
DATASET_REPO = "local/smolvla"
POLICY = "lerobot/smolvla_base"
RESULTS_DIR = BENCH_DIR
BASELINE_DIR = BENCH_DIR / "baseline"
OURS_DIR = BENCH_DIR / "ours"
BATCH_SIZE = 4
STEPS = 8
NUM_WORKERS = 2
LOG_FREQ = 1
RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
METRIC_RE = re.compile(
    r"(step|smpl|ep|epch|loss|grdn|lr|updt_s|data_s|smp/s|mem_gb):([0-9.+\-eE]+[KMB]?)"
)
SCRIPT_DEADLINE_S = 540
# ======Settings=========


def _parse_number(raw: str) -> float:
    text = str(raw or "").strip()
    factor = 1.0
    if text.endswith(("K", "M", "B")):
        factor = {"K": 1e3, "M": 1e6, "B": 1e9}[text[-1]]
        text = text[:-1]
    return float(text) * factor


def parse_metrics(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if "loss:" not in line or "step:" not in line:
            continue
        found = {key: _parse_number(value) for key, value in METRIC_RE.findall(line)}
        if "step" in found and "loss" in found:
            rows.append(found)
    return rows


def summarize(rows: list[dict], elapsed_s: float, label: str) -> dict:
    if not rows:
        return {
            "label": label,
            "ok": False,
            "elapsed_s": elapsed_s,
            "steps": 0,
        }
    skip = 1 if len(rows) > 2 else 0
    steady = rows[skip:]
    def mean(key: str) -> float | None:
        values = [row[key] for row in steady if key in row]
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "label": label,
        "ok": True,
        "elapsed_s": elapsed_s,
        "steps": int(rows[-1]["step"]),
        "loss_last": rows[-1]["loss"],
        "updt_s": mean("updt_s"),
        "data_s": mean("data_s"),
        "smp_s": mean("smp/s"),
        "mem_gb": mean("mem_gb"),
        "rows": rows,
    }


def common_args(output: Path, use_amp: bool, empty_cameras: int) -> list[str]:
    return [
        f"--policy.path={POLICY}",
        f"--dataset.repo_id={DATASET_REPO}",
        f"--dataset.root={stored_path(DATASET_ROOT)}",
        f"--output_dir={stored_path(output)}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        "--job_name=bench",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--rename_map={json.dumps(RENAME_MAP, separators=(',', ':'))}",
        f"--policy.empty_cameras={empty_cameras}",
        "--save_checkpoint=false",
        "--save_freq=0",
        f"--num_workers={NUM_WORKERS}",
        f"--log_freq={LOG_FREQ}",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
        f"--policy.use_amp={'true' if use_amp else 'false'}",
    ]


def run_command(
    module: str,
    output: Path,
    use_amp: bool,
    empty_cameras: int = 1,
) -> dict:
    if output.exists():
        shutil.rmtree(output)
    command = [
        sys.executable,
        "-m",
        module,
        *common_args(output, use_amp, empty_cameras),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=SCRIPT_DEADLINE_S,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - started
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary = summarize(parse_metrics(text), elapsed, module)
    summary["returncode"] = proc.returncode
    summary["use_amp"] = use_amp
    if proc.returncode != 0:
        summary["ok"] = False
        summary["tail"] = text[-4000:]
    return summary


def check_variable_cameras() -> dict:
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_policy, make_pre_post_processors

    from train_loop.cameras import VariableCameraDataset, variable_camera_collate

    started = time.perf_counter()
    cfg = PreTrainedConfig.from_pretrained(POLICY)
    cfg.device = "cuda"
    cfg.pretrained_path = POLICY
    meta = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    dataset = LeRobotDataset(
        DATASET_REPO,
        root=DATASET_ROOT,
        return_uint8=True,
        delta_timestamps=resolve_delta_timestamps(cfg, meta),
    )
    present = {}
    for index in range(int(dataset.num_episodes)):
        if index % 2 == 0:
            present[index] = ["observation.images.front"]
        else:
            present[index] = [
                "observation.images.front",
                "observation.images.wrist",
            ]
    wrapped = VariableCameraDataset(dataset, present, cache_frames=False)
    one = None
    two = None
    for index in range(len(wrapped)):
        item = wrapped[index]
        cameras = sorted(
            key for key in item if key.startswith("observation.images.")
        )
        if cameras == ["observation.images.front"]:
            one = item
        elif "observation.images.wrist" in cameras:
            two = item
        if one is not None and two is not None:
            break
    if one is None or two is None:
        raise RuntimeError("could not find mixed-camera samples")
    mixed = variable_camera_collate([one, two])
    mixed_cams = sorted(
        key for key in mixed if key.startswith("observation.images.")
    )
    if mixed_cams != ["observation.images.front"]:
        raise RuntimeError(f"collate kept extra cameras: {mixed_cams}")
    same = variable_camera_collate([two, dict(two)])
    same_cams = sorted(
        key for key in same if key.startswith("observation.images.")
    )
    if "observation.images.wrist" not in same_cams:
        raise RuntimeError(f"same-set collate lost wrist: {same_cams}")

    policy = make_policy(
        cfg=cfg,
        ds_meta=dataset.meta,
        rename_map=RENAME_MAP,
    )
    preprocessor, _post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=POLICY,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    policy.cuda().train()
    for key, value in list(mixed.items()):
        if torch.is_tensor(value) and value.dtype == torch.uint8:
            mixed[key] = value.float() / 255.0
    batch = preprocessor(mixed)
    loss, _ = policy.forward(batch)
    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "elapsed_s": elapsed,
        "mixed_cameras": mixed_cams,
        "same_cameras": same_cams,
        "loss": float(loss.detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("baseline", "ours", "both", "variable", "empty0", "mixed"),
        default="baseline",
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()
    if args.steps > 0:
        global STEPS
        STEPS = args.steps
    if args.batch > 0:
        global BATCH_SIZE
        BATCH_SIZE = args.batch
    ensure_data_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_ROOT.is_dir():
        raise SystemExit(f"dataset missing: {DATASET_ROOT}")

    payload: dict = {"mode": args.mode, "batch": BATCH_SIZE, "steps": STEPS}
    if args.mode in ("baseline", "both"):
        payload["baseline"] = run_command(
            "lerobot.scripts.lerobot_train",
            BASELINE_DIR,
            use_amp=False,
        )
        (RESULTS_DIR / "baseline.json").write_text(
            json.dumps(payload["baseline"], indent=2)
        )
        print(json.dumps(payload["baseline"], indent=2))
    if args.mode in ("ours", "both"):
        payload["ours"] = run_command("train_loop.lerobot_train", OURS_DIR, use_amp=True)
        (RESULTS_DIR / "ours.json").write_text(json.dumps(payload["ours"], indent=2))
        print(json.dumps(payload["ours"], indent=2))
    if args.mode == "variable":
        payload["variable"] = check_variable_cameras()
        (RESULTS_DIR / "variable.json").write_text(
            json.dumps(payload["variable"], indent=2)
        )
        print(json.dumps(payload["variable"], indent=2))
    if args.mode == "mixed":
        sidecar = DATASET_ROOT / "meta" / "episode_cameras.json"
        backup = sidecar.read_text() if sidecar.is_file() else None
        mixed = {
            str(index): (
                ["observation.images.front"]
                if index % 2 == 0
                else ["observation.images.front", "observation.images.wrist"]
            )
            for index in range(9)
        }
        sidecar.write_text(json.dumps(mixed, indent=2))
        try:
            payload["mixed"] = run_command(
                "train_loop.lerobot_train",
                BENCH_DIR / "mixed",
                use_amp=True,
                empty_cameras=0,
            )
        finally:
            if backup is None:
                sidecar.unlink(missing_ok=True)
            else:
                sidecar.write_text(backup)
        (RESULTS_DIR / "mixed.json").write_text(json.dumps(payload["mixed"], indent=2))
        print(json.dumps(payload["mixed"], indent=2))
    if args.mode == "empty0":
        payload["empty0"] = run_command(
            "train_loop.lerobot_train",
            BENCH_DIR / "empty0",
            use_amp=True,
            empty_cameras=0,
        )
        (RESULTS_DIR / "empty0.json").write_text(
            json.dumps(payload["empty0"], indent=2)
        )
        print(json.dumps(payload["empty0"], indent=2))
    (RESULTS_DIR / f"{args.mode}.json").write_text(json.dumps(payload, indent=2))
    failed = [
        name
        for name in ("baseline", "ours", "variable", "empty0", "mixed")
        if name in payload and not payload[name].get("ok")
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
