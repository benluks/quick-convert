"""Split manifest rows within groups or hold out complete groups."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a manifest between training and validation.",
    )
    parser.add_argument("--input", required=True, help="Input manifest CSV")
    parser.add_argument("--train-output", required=True, help="Output training manifest CSV")
    parser.add_argument("--valid-output", required=True, help="Output validation manifest CSV")
    parser.add_argument("--group-col", default="spkid", help="Column used for grouping (default: spkid)")
    parser.add_argument(
        "--strategy",
        choices=["within-group", "group-disjoint"],
        default="within-group",
        help=(
            "Split rows within every group, or assign complete groups to one partition "
            "(default: within-group)."
        ),
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="Validation fraction of rows or groups, depending on strategy (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Manifest splitting requires the manifests extra.") from error

    from quick_convert.data.splits import (
        split_manifest_group_disjoint,
        split_manifest_within_groups,
    )

    args = parse_args()
    manifest = pd.read_csv(args.input)

    split_fn = {
        "within-group": split_manifest_within_groups,
        "group-disjoint": split_manifest_group_disjoint,
    }[args.strategy]
    train, valid = split_fn(
        manifest,
        group_col=args.group_col,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )

    train_path = pd.io.common.stringify_path(args.train_output)
    valid_path = pd.io.common.stringify_path(args.valid_output)
    from pathlib import Path

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(valid_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)


if __name__ == "__main__":
    main()
