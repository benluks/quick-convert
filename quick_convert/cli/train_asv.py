from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quick_convert.asv.train import train_asv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an ASV model using the SpeechBrain-based recipe."
    )
    parser.add_argument(
        "hparams",
        type=Path,
        help="Path to the hyperparameter YAML file.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional HyperPyYAML / SpeechBrain overrides, e.g. "
            "data_folder=/path/to/data output_folder=exp/asv"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    hparams_path = args.hparams.expanduser().resolve()
    if not hparams_path.exists():
        parser.error(f"Hyperparameter file does not exist: {hparams_path}")

    train_asv(
        hparams_file=hparams_path,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
