#!/usr/bin/env python
"""LeRobot-compatible training CLI. Same flags as `lerobot.scripts.lerobot_train`."""

import logging
import sys

from lerobot.configs import JobConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins

from train_loop.loop import train as run_train


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    return run_train(cfg)


def _remote_target_in_argv() -> bool:
    target = None
    args = sys.argv[1:]
    for i, tok in enumerate(args):
        if tok == "--job.target" and i + 1 < len(args):
            target = args[i + 1]
        elif tok.startswith("--job.target="):
            target = tok.split("=", 1)[1]
    return JobConfig.is_remote_target(target)


def main():
    register_third_party_plugins()
    if _remote_target_in_argv():
        logging.getLogger("lerobot.configs.policies").setLevel(logging.ERROR)
    train()


if __name__ == "__main__":
    main()
