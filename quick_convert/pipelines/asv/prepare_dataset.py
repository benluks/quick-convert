from __future__ import annotations

import csv
import random
from pathlib import Path

import torchaudio


from collections import defaultdict
from ...data.base_dataset import AudioSample


def split_rows_by_seen_speakers(rows, train_percent, seed=1337):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)

    rng = random.Random(seed)

    train_rows = []
    heldout_rows = []

    for split, split_rows in by_split.items():
        split_rows = sorted(split_rows, key=lambda r: r.spk_id)
        rng.shuffle(split_rows)
        n_train = int(len(split_rows) * train_percent)

        train_rows.extend(split_rows[:n_train])
        heldout_rows.extend(split_rows[n_train:])

    return train_rows, heldout_rows


def write_sb_csv(rows: list[AudioSample], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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


def prepare_asv_csvs(
    data_folder: str,
    save_folder: str,
    splits: list[str] | None = None,
    split_ratio: list[int] = [90, 10],
    seg_dur: float = 3.0,
    skip_prep: bool = False,
    random_segment: bool = False,
    file_pattern: str = "*.wav",
    spkid_fn: str | None = None,
):
    if skip_prep:
        return

    data_root = Path(data_folder)
    save_root = Path(save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    train_csv = save_root / "train.csv"
    dev_csv = save_root / "dev.csv"

    if train_csv.exists() and dev_csv.exists():
        return

    # Collect files
    search_roots = [data_root / s for s in splits] if splits else [data_root]
    audio_files: list[Path] = []
    for root in search_roots:
        audio_files.extend(root.rglob(file_pattern))

    # Speaker ID logic:
    # adapt this to your dataset
    # here I assume speaker is the immediate parent dir
    rows = []
    for wav_path in sorted(audio_files):
        try:
            info = torchaudio.info(str(wav_path))
        except Exception:
            continue

        num_frames = info.num_frames
        sr = info.sample_rate
        duration = num_frames / sr

        # custom spkid function
        spk_id = spkid_fn(wav_path)

        rows.append(
            {
                "ID": wav_path.stem,
                "duration": duration,
                "wav": str(wav_path),
                "start": 0,
                "stop": num_frames,
                "spk_id": spk_id,
            }
        )

    random.shuffle(rows)
    split_idx = int(len(rows) * split_ratio[0] / 100)
    train_rows = rows[:split_idx]
    dev_rows = rows[split_idx:]

    for path, subset in [(train_csv, train_rows), (dev_csv, dev_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "duration", "wav", "start", "stop", "spk_id"])
            for row in subset:
                writer.writerow(
                    [
                        row["ID"],
                        row["duration"],
                        row["wav"],
                        row["start"],
                        row["stop"],
                        row["spk_id"],
                    ]
                )
