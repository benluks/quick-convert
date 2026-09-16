from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from quick_convert.inference import export_inference_artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quick-convert export",
        description="Export a training run for inference.",
    )
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument("destination", type=Path, help="Output directory")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/last.ckpt",
        help="Checkpoint path, relative to the run directory by default",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Resolved config path, relative to the run directory by default",
    )
    parser.add_argument("--device", default="cpu", help="Device used while reading the checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty destination")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parser().parse_args(argv)
    destination = export_inference_artifact(
        args.run_dir,
        args.destination,
        checkpoint=args.checkpoint,
        config=args.config,
        map_location=args.device,
        overwrite=args.overwrite,
    )
    print(destination)
    return destination
