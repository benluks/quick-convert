from __future__ import annotations

import csv
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torchaudio
from tqdm import tqdm

from quick_convert.data.base_dataset import AudioSample, BaseDataset


def _get_audio_info(row: AudioSample):
    info = torchaudio.info(str(row.path))
    return {
        "row": row,
        "num_frames": info.num_frames,
        "sample_rate": info.sample_rate,
        "duration": info.num_frames / info.sample_rate if info.sample_rate > 0 else 0.0,
    }


def prepare_asv_csvs_from_dataset(
    dataset: BaseDataset,
    save_folder: str | Path,
    train_fraction: float = 0.9,
    seed: int = 1337,
    randomize_within_split: bool = False,
    num_workers: int = 8,
) -> tuple[str, str, int]:
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    train_csv = save_folder / "train.csv"
    dev_csv = save_folder / "dev.csv"

    rows = list(dataset)

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

    def collect_metadata(rows: list[AudioSample]) -> list[dict]:
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_get_audio_info, row) for row in rows]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Reading audio metadata",
            ):
                results.append(fut.result())

        # Preserve original row order
        row_to_index = {id(row): i for i, row in enumerate(rows)}
        results.sort(key=lambda x: row_to_index[id(x["row"])])
        return results

    def write_csv(path: Path, rows: list[AudioSample]) -> None:
        metadata = collect_metadata(rows)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["ID", "duration", "sample_rate", "wav", "start", "stop", "spk_id"]
            )

            out_idx = 0
            for item in tqdm(metadata, total=len(metadata), desc=f"Writing {path}"):
                row = item["row"]
                num_frames = item["num_frames"]
                sample_rate = item["sample_rate"]
                duration = item["duration"]

                if num_frames <= 0:
                    print(f"Skipping zero-length file: {row.path}")
                    continue
                if sample_rate <= 0:
                    print(f"Skipping invalid sample-rate file: {row.path}")
                    continue

                writer.writerow(
                    [
                        str(out_idx),
                        duration,
                        sample_rate,
                        str(row.path),
                        0,
                        num_frames,
                        row.spk_id,
                    ]
                )
                out_idx += 1

    write_csv(train_csv, train_rows)
    write_csv(dev_csv, dev_rows)

    n_speakers = len({row.spk_id for row in train_rows})

    return str(train_csv), str(dev_csv), n_speakers