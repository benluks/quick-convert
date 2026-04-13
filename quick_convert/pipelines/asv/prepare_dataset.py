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


from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def prepare_asv_eval_data(
    input_csv: str | Path,
    output_dir: str | Path,
    enrol_splits: Iterable[str],
    test_splits: Iterable[str],
    *,
    overwrite: bool = False,
    negatives_per_enrol: int | None = 10,
    seed: int = 1337,
) -> tuple[str, str, str]:
    """
    Prepare SpeechBrain-style enrol.csv, test.csv, and trials.txt from a master CSV.

    Assumptions:
    - `wav` paths look like /.../<split>/<filename>.wav
    - the parent folder name determines the split
    - `spk_id` identifies the speaker
    - the same speakers appear across multiple splits

    Trials format:
        <label> <enrol_id> <test_id>
    where label is:
        1 = same speaker
        0 = different speaker

    Args:
        input_csv:
            Path to the master SpeechBrain-style CSV.
        output_dir:
            Output directory for enrol.csv, test.csv, trials.txt
        enrol_splits:
            Parent folder names assigned to the enrol set.
        test_splits:
            Parent folder names assigned to the test set.
        overwrite:
            Whether to overwrite existing files.
        negatives_per_enrol:
            Number of negative trials to generate per enrol utterance.
            If None, generate all possible impostor pairings.
        seed:
            Random seed for negative sampling.

    Returns:
        (enrol_csv_path, test_csv_path, trials_txt_path)
    """
    rng = random.Random(seed)

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enrol_csv = output_dir / "enrol.csv"
    test_csv = output_dir / "test.csv"
    trials_txt = output_dir / "trials.txt"

    if (
        not overwrite
        and enrol_csv.is_file()
        and test_csv.is_file()
        and trials_txt.is_file()
    ):
        return str(enrol_csv), str(test_csv), str(trials_txt)

    enrol_splits = set(enrol_splits)
    test_splits = set(test_splits)

    overlap = enrol_splits & test_splits
    if overlap:
        raise ValueError(
            f"These splits are assigned to both enrol and test: {sorted(overlap)}"
        )

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in CSV: {input_csv}")

        enrol_rows: list[dict[str, str]] = []
        test_rows: list[dict[str, str]] = []
        unassigned_splits: set[str] = set()

        for row in reader:
            split_name = Path(row["wav"]).parent.name

            if split_name in enrol_splits:
                enrol_rows.append(row)
            elif split_name in test_splits:
                test_rows.append(row)
            else:
                unassigned_splits.add(split_name)

    if unassigned_splits:
        raise ValueError(
            "Found rows whose split was not assigned to enrol or test: "
            f"{sorted(unassigned_splits)}"
        )

    if not enrol_rows:
        raise ValueError("No rows assigned to enrol")
    if not test_rows:
        raise ValueError("No rows assigned to test")

    # Write enrol/test CSVs
    with enrol_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enrol_rows)

    with test_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    # Index test utterances by speaker
    test_by_spk: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        test_by_spk[row["spk_id"]].append(row)

    all_test_rows_by_other_spk: dict[str, list[dict[str, str]]] = {}
    all_speakers_in_test = sorted(test_by_spk.keys())

    for spk in all_speakers_in_test:
        impostors = []
        for other_spk in all_speakers_in_test:
            if other_spk == spk:
                continue
            impostors.extend(test_by_spk[other_spk])
        all_test_rows_by_other_spk[spk] = impostors

    # Generate trials
    n_positive = 0
    n_negative = 0

    with trials_txt.open("w", encoding="utf-8") as f:
        for enrol_row in enrol_rows:
            enrol_id = enrol_row["ID"]
            enrol_spk = enrol_row["spk_id"]

            positive_candidates = test_by_spk.get(enrol_spk, [])
            if not positive_candidates:
                raise ValueError(
                    f"No test utterances found for enrol speaker {enrol_spk}"
                )

            # Positive trials: enrol utterance against all test utterances
            # from the same speaker in the test set.
            for test_row in positive_candidates:
                test_id = test_row["ID"]
                f.write(f"1 {enrol_id} {test_id}\n")
                n_positive += 1

            # Negative trials
            negative_candidates = all_test_rows_by_other_spk[enrol_spk]
            if not negative_candidates:
                raise ValueError(
                    f"No impostor test utterances found for enrol speaker {enrol_spk}"
                )

            if negatives_per_enrol is None:
                sampled_negatives = negative_candidates
            else:
                k = min(negatives_per_enrol, len(negative_candidates))
                sampled_negatives = rng.sample(negative_candidates, k)

            for test_row in sampled_negatives:
                test_id = test_row["ID"]
                f.write(f"0 {enrol_id} {test_id}\n")
                n_negative += 1

    print(
        f"Wrote {enrol_csv}, {test_csv}, {trials_txt} "
        f"with {len(enrol_rows)} enrol rows, {len(test_rows)} test rows, "
        f"{n_positive} positive trials, {n_negative} negative trials."
    )

    return str(enrol_csv), str(test_csv), str(trials_txt)
