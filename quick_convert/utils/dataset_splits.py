"""
This is kind of a throwaway, because I can't think of how to abstract it.
"""

import pandas as pd


def split_manifest_by_group(
    df,
    group_col="spkid",
    valid_fraction=0.1,
    seed=0,
):
    valid_parts = []
    train_parts = []

    for _, group in df.groupby(group_col):
        valid = group.sample(frac=valid_fraction, random_state=seed)

        # ensure at least one validation item if group has >1 utterance
        if len(valid) == 0 and len(group) > 1:
            valid = group.sample(n=1, random_state=seed)

        train = group.drop(valid.index)

        valid_parts.append(valid)
        train_parts.append(train)

    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed),
        pd.concat(valid_parts).sample(frac=1, random_state=seed),
    )


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split a manifest into train/validation sets while preserving groups.")

    parser.add_argument(
        "--input",
        required=True,
        help="Input manifest CSV",
    )

    parser.add_argument(
        "--train-output",
        required=True,
        help="Output train manifest CSV",
    )

    parser.add_argument(
        "--valid-output",
        required=True,
        help="Output validation manifest CSV",
    )

    parser.add_argument(
        "--group-col",
        default="spkid",
        help="Column used for grouping (default: spkid)",
    )

    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="Fraction of each group to place in validation (default: 0.1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input)
    train_df, valid_df = split_manifest_by_group(
        df,
        group_col=args.group_col,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )

    train_df.to_csv(args.train_output, index=False)
    valid_df.to_csv(args.valid_output, index=False)
