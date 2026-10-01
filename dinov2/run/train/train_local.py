# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Local single-GPU launcher for DINOv2 training.

Unlike ``dinov2/run/train/train.py``, this launcher does NOT use submitit /
SLURM. It runs the training loop directly in the current process, which is what
you want on a single workstation GPU.

``dinov2.train.main`` internally calls ``dinov2.distributed.enable()``, which
already has a single-GPU fallback: with no SLURM / torchrun environment set it
initializes a one-process group (rank 0, world_size 1) on 127.0.0.1 and calls
``torch.cuda.set_device(0)``. So no distributed launcher is required.

Usage:
    python -m dinov2.run.train.train_local \
        --config-file dinov2/configs/train/cle_vits.yaml \
        --output-dir /path/to/output

You can override any config value on the command line, e.g.:
    ... train.batch_size_per_gpu=16 student.arch=vit_small
"""

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.train import main as train_main


logger = logging.getLogger("dinov2")


def main():
    description = "Local (single-GPU, no submitit) launcher for DINOv2 training"
    args_parser = get_train_args_parser(add_help=True)
    args_parser.description = description
    args = args_parser.parse_args()

    setup_logging()

    assert args.config_file and os.path.exists(args.config_file), (
        "Configuration file does not exist! Pass a valid --config-file."
    )
    if not args.output_dir:
        raise SystemExit("Please pass --output-dir <path> to store logs and checkpoints.")

    logger.info("Running DINOv2 training locally (single process, no submitit).")
    train_main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
