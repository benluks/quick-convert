"""Split each speaker's manifest rows between training and validation."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split each manifest group between training and validation.",
    )
    parser.add_argument("--input", required=True, help="Input manifest CSV")
    parser.add_argument("--train-output", required=True, help="Output training manifest CSV")
    parser.add_argument("--valid-output", required=True, help="Output validation manifest CSV")
    parser.add_argument("--group-col", default="spkid", help="Column used for grouping (default: spkid)")
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="Fraction of each group's rows assigned to validation (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Manifest splitting requires the `manifests` extra.") from error

    from quick_convert.data.splits import split_manifest_within_groups

    args = parse_args()
    manifest = pd.read_csv(args.input)
    train, valid = split_manifest_within_groups(
        manifest,
        group_col=args.group_col,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    train.to_csv(args.train_output, index=False)
    valid.to_csv(args.valid_output, index=False)


if __name__ == "__main__":
    main()
