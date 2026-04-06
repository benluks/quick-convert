from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torchaudio

from quick_convert.data.base_dataset import AudioSample, BaseDataset


def prepare_asv_csvs_from_dataset(
    dataset: BaseDataset,
    save_folder: str | Path,
    train_fraction: float = 0.9,
    seed: int = 1337,
    randomize_within_split: bool = False,
) -> tuple[str, str, int]:
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    train_csv = save_folder / "train.csv"
    dev_csv = save_folder / "dev.csv"

    rows = list(dataset)

    # Require speaker IDs
    missing = [row.path for row in rows if row.spk_id is None]
    if missing:
        raise ValueError(
            "Some dataset rows are missing spk_id. "
            "Make sure return_spkid=True and get_spkid() is implemented."
        )

    by_split = defaultdict(list)
    for row in rows:
        split = row.split or ""
        by_split[split].append(row)

    rng = random.Random(seed)

    train_rows: list[AudioSample] = []
    dev_rows: list[AudioSample] = []

    for split, split_rows in by_split.items():
        split_rows = sorted(split_rows, key=lambda r: r.spk_id)
        if randomize_within_split:
            rng.shuffle(split_rows)

        n_train = int(len(split_rows) * train_fraction)
        train_rows.extend(split_rows[:n_train])
        dev_rows.extend(split_rows[n_train:])

    def write_csv(path: Path, rows: list[AudioSample]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "duration", "wav", "start", "stop", "spk_id"])

            for row in rows:
                info = torchaudio.info(str(row.path))
                duration = info.num_frames / info.sample_rate

                writer.writerow(
                    [
                        row.path.stem,
                        duration,
                        str(row.path),
                        0,
                        info.num_frames,
                        row.spk_id,
                    ]
                )

    write_csv(train_csv, train_rows)
    write_csv(dev_csv, dev_rows)

    n_speakers = len({row.spk_id for row in train_rows})

    return str(train_csv), str(dev_csv), n_speakers