from __future__ import annotations

import csv
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import torchaudio
from tqdm import tqdm

from quick_convert.data.base_dataset import AudioSample, BaseDataset


def _get_audio_info(row: AudioSample) -> dict:
    info = torchaudio.info(str(row.path))
    return {
        "row": row,
        "num_frames": info.num_frames,
        "sample_rate": info.sample_rate,
        "duration": info.num_frames / info.sample_rate if info.sample_rate > 0 else 0.0,
    }


def _collect_audio_metadata(
    rows: list[AudioSample],
    *,
    num_workers: int = 8,
) -> list[dict]:
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_get_audio_info, row) for row in rows]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Reading audio metadata",
        ):
            results.append(fut.result())

    row_to_index = {id(row): i for i, row in enumerate(rows)}
    results.sort(key=lambda x: row_to_index[id(x["row"])])
    return results


def _assert_no_speaker_overlap(
    train_rows: list[AudioSample],
    dev_rows: list[AudioSample],
) -> None:
    train_speakers = {row.spk_id for row in train_rows}
    dev_speakers = {row.spk_id for row in dev_rows}
    overlap = train_speakers & dev_speakers
    if overlap:
        raise ValueError(
            f"Speaker overlap detected between train and dev: "
            f"{len(overlap)} speakers: {sorted(overlap)}"
        )


def _write_audio_sample_csv(
    path: Path,
    rows: list[AudioSample],
    *,
    num_workers: int = 8,
) -> None:
    metadata = _collect_audio_metadata(rows, num_workers=num_workers)

    with path.open("w", newline="", encoding="utf-8") as f:
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


def _resolve_eval_paths(output_dir: str | Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return (
        output_dir / "enrol.csv",
        output_dir / "test.csv",
        output_dir / "trials.txt",
    )


def _load_csv_rows(input_csv: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    input_csv = Path(input_csv)

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"No header found in CSV: {input_csv}")
        rows = list(reader)

    return rows, fieldnames


def _write_dict_rows_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_impostor_index(
    test_by_spk: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    all_speakers = sorted(test_by_spk.keys())
    impostors_by_spk: dict[str, list[dict[str, str]]] = {}

    for spk in all_speakers:
        impostors: list[dict[str, str]] = []
        for other_spk in all_speakers:
            if other_spk == spk:
                continue
            impostors.extend(test_by_spk[other_spk])
        impostors_by_spk[spk] = impostors

    return impostors_by_spk


def _write_trials(
    trials_txt: Path,
    enrol_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    *,
    negatives_per_enrol: int | None = 10,
    seed: int = 1337,
) -> tuple[int, int]:
    rng = random.Random(seed)

    test_by_spk: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        test_by_spk[row["spk_id"]].append(row)

    impostors_by_spk = _build_impostor_index(test_by_spk)

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

            for test_row in positive_candidates:
                f.write(f"1 {enrol_id} {test_row['ID']}\n")
                n_positive += 1

            negative_candidates = impostors_by_spk.get(enrol_spk, [])
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
                f.write(f"0 {enrol_id} {test_row['ID']}\n")
                n_negative += 1

    return n_positive, n_negative


def _finalize_eval_data(
    *,
    fieldnames: list[str],
    enrol_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    enrol_csv: Path,
    test_csv: Path,
    trials_txt: Path,
    negatives_per_enrol: int | None = 10,
    seed: int = 1337,
) -> tuple[str, str, str]:
    if not enrol_rows:
        raise ValueError("No rows assigned to enrol")
    if not test_rows:
        raise ValueError("No rows assigned to test")

    _write_dict_rows_csv(enrol_csv, fieldnames, enrol_rows)
    _write_dict_rows_csv(test_csv, fieldnames, test_rows)

    n_positive, n_negative = _write_trials(
        trials_txt,
        enrol_rows,
        test_rows,
        negatives_per_enrol=negatives_per_enrol,
        seed=seed,
    )

    print(
        f"Wrote {enrol_csv}, {test_csv}, {trials_txt} "
        f"with {len(enrol_rows)} enrol rows, {len(test_rows)} test rows, "
        f"{n_positive} positive trials, {n_negative} negative trials."
    )

    return str(enrol_csv), str(test_csv), str(trials_txt)


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

    by_spk: dict[str, list[AudioSample]] = defaultdict(list)
    for row in rows:
        by_spk[row.spk_id].append(row)

    speakers = sorted(by_spk.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n_train = int(len(speakers) * train_fraction)
    train_speakers = set(speakers[:n_train])
    dev_speakers = set(speakers[n_train:])

    train_rows: list[AudioSample] = []
    dev_rows: list[AudioSample] = []

    for spk, spk_rows in by_spk.items():
        spk_rows = list(spk_rows)
        if randomize_within_split:
            rng.shuffle(spk_rows)

        if spk in train_speakers:
            train_rows.extend(spk_rows)
        elif spk in dev_speakers:
            dev_rows.extend(spk_rows)
        else:
            raise RuntimeError(f"Speaker {spk} was assigned to neither train nor dev")

    train_overlap = {row.spk_id for row in train_rows} & {
        row.spk_id for row in dev_rows
    }
    if train_overlap:
        raise ValueError(
            f"Speaker overlap detected between train and dev: {sorted(train_overlap)}"
        )

    _assert_no_speaker_overlap(train_rows, dev_rows)

    _write_audio_sample_csv(train_csv, train_rows, num_workers=num_workers)
    _write_audio_sample_csv(dev_csv, dev_rows, num_workers=num_workers)

    n_speakers = len(train_speakers)

    return str(train_csv), str(dev_csv), n_speakers


def prepare_asv_eval_by_split(
    input_csv: str | Path,
    output_dir: str | Path,
    enrol_splits: Iterable[str],
    test_splits: Iterable[str],
    *,
    overwrite: bool = False,
    negatives_per_enrol: int | None = 10,
    seed: int = 1337,
) -> tuple[str, str, str]:
    enrol_csv, test_csv, trials_txt = _resolve_eval_paths(output_dir)

    if (
        not overwrite
        and enrol_csv.is_file()
        and test_csv.is_file()
        and trials_txt.is_file()
    ):
        return str(enrol_csv), str(test_csv), str(trials_txt)

    rows, fieldnames = _load_csv_rows(input_csv)

    enrol_splits = set(enrol_splits)
    test_splits = set(test_splits)

    overlap = enrol_splits & test_splits
    if overlap:
        raise ValueError(
            f"These splits are assigned to both enrol and test: {sorted(overlap)}"
        )

    enrol_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    unassigned_splits: set[str] = set()

    for row in rows:
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

    return _finalize_eval_data(
        fieldnames=fieldnames,
        enrol_rows=enrol_rows,
        test_rows=test_rows,
        enrol_csv=enrol_csv,
        test_csv=test_csv,
        trials_txt=trials_txt,
        negatives_per_enrol=negatives_per_enrol,
        seed=seed,
    )


def prepare_asv_eval_random(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    enrol_per_speaker: int = 1,
    negatives_per_enrol: int | None = 10,
    seed: int = 1337,
    overwrite: bool = False,
) -> tuple[str, str, str]:
    rng = random.Random(seed)

    enrol_csv, test_csv, trials_txt = _resolve_eval_paths(output_dir)

    if (
        not overwrite
        and enrol_csv.is_file()
        and test_csv.is_file()
        and trials_txt.is_file()
    ):
        return str(enrol_csv), str(test_csv), str(trials_txt)

    rows, fieldnames = _load_csv_rows(input_csv)

    by_spk: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_spk[row["spk_id"]].append(row)

    enrol_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for spk, spk_rows in by_spk.items():
        if len(spk_rows) <= enrol_per_speaker:
            raise ValueError(f"Speaker {spk} has too few utterances ({len(spk_rows)})")

        spk_rows = list(spk_rows)
        rng.shuffle(spk_rows)

        enrol_rows.extend(spk_rows[:enrol_per_speaker])
        test_rows.extend(spk_rows[enrol_per_speaker:])

    return _finalize_eval_data(
        fieldnames=fieldnames,
        enrol_rows=enrol_rows,
        test_rows=test_rows,
        enrol_csv=enrol_csv,
        test_csv=test_csv,
        trials_txt=trials_txt,
        negatives_per_enrol=negatives_per_enrol,
        seed=seed,
    )


def prepare_asv_eval_data(mode, **kwargs) -> tuple[str, str, str]:
    if mode == "by_split":
        return prepare_asv_eval_by_split(**kwargs)
    elif mode == "random":
        return prepare_asv_eval_random(**kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
