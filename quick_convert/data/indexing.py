"""Compose recording metadata and annotations into dataset rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from pathlib import Path

from .resources import ResourceCollection, ResourceRef
from .types import MetadataSample


def read_csv_index(path: str | Path, *, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read the requested CSV columns as strings, preserving row order."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if any(header.count(column) != 1 for column in columns):
            raise ValueError(f"Requested CSV columns must each appear once in {path}.")
        result = []
        for row in reader:
            if any(row[column] is None for column in columns):
                raise ValueError(f"Incomplete CSV row at {path}:{reader.line_num}.")
            result.append({column: row[column] for column in columns})
        return result


def _join_key(row: Mapping[str, str], columns: Sequence[str]) -> tuple[str, ...]:
    key = tuple(row[column] for column in columns)
    if any(not value.strip() for value in key):
        raise ValueError(f"Index key columns must be nonempty: {list(columns)}.")
    return key


def join_index(
    rows: Iterable[Mapping[str, str]],
    lookup: Iterable[Mapping[str, str]],
    *,
    key_columns: Sequence[str],
) -> list[dict[str, str]]:
    """Join every input row to exactly one lookup row on shared key columns."""
    if not key_columns:
        raise ValueError("Supply at least one index join column.")
    by_key = {}
    for row in lookup:
        key = _join_key(row, key_columns)
        if key in by_key:
            raise ValueError(f"Duplicate lookup key {key!r}.")
        by_key[key] = row

    result = []
    for row in rows:
        key = _join_key(row, key_columns)
        if key not in by_key:
            raise ValueError(f"No lookup row for key {key!r}.")
        other = by_key[key]
        for column in row.keys() & other.keys():
            if row[column] != other[column]:
                raise ValueError(f"Conflicting values for index column {column!r} at key {key!r}.")
        result.append({**row, **other})
    return result


def link_identities(
    rows: Iterable[Mapping[str, str]],
    *,
    id_column: str,
    links_column: str,
    output_column: str,
    separator: str,
) -> list[dict[str, str]]:
    """Group linked IDs transitively and assign each group its smallest ID.

    Links may point in either direction. A blank link cell adds no links.
    Each referenced ID must identify a unique row in this index.
    """
    if not separator:
        raise ValueError("Identity link separator must be nonempty.")
    rows = list(rows)
    identities = set()
    neighbors = defaultdict(set)
    for row in rows:
        (identity,) = _join_key(row, (id_column,))
        if identity in identities:
            raise ValueError(f"Duplicate identity {identity!r}.")
        if output_column in row:
            raise ValueError(f"Identity output column {output_column!r} already exists.")
        identities.add(identity)
        for value in row[links_column].split(separator):
            linked = value.strip()
            if linked:
                neighbors[identity].add(linked)
                neighbors[linked].add(identity)

    unknown = neighbors.keys() - identities
    if unknown:
        raise ValueError(f"Unknown linked identity {min(unknown)!r}.")

    groups = {}
    for identity in sorted(identities):
        if identity in groups:
            continue
        groups[identity] = identity
        pending = [identity]
        while pending:
            for linked in neighbors[pending.pop()]:
                if linked not in groups:
                    groups[linked] = identity
                    pending.append(linked)
    return [{**row, output_column: groups[row[id_column]]} for row in rows]


def _rttm_intervals(path: Path, speaker: str) -> list[tuple[float, float]]:
    intervals = []
    recording_channel = None
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] != "SPEAKER":
                continue
            if len(fields) < 8:
                raise ValueError(f"Incomplete RTTM speaker record at {path}:{line_no}.")
            current = (fields[1], fields[2])
            if recording_channel is not None and recording_channel != current:
                raise ValueError(f"Expected one recording and channel in {path}.")
            recording_channel = current
            try:
                start, duration = map(float, fields[3:5])
            except ValueError as error:
                raise ValueError(f"Invalid RTTM interval at {path}:{line_no}.") from error
            end = start + duration
            if not isfinite(start) or not isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"Invalid RTTM interval at {path}:{line_no}.")
            if fields[7] == speaker:
                intervals.append((start, end))
    return sorted(intervals)


def rttm_segment_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    recording_id_column: str,
    audio_path: str,
    rttm_path: str,
    speaker: str,
    resources: Mapping[str, str] | None = None,
) -> list[MetadataSample]:
    """Build one metadata row per selected RTTM turn without decoding audio.

    The path and speaker arguments are format strings evaluated against each
    recording row. Resource names map to columns whose values are attached as
    text resources. Recordings without the selected label contribute no rows.
    """
    result = []
    recording_ids = set()
    for row in rows:
        (recording_id,) = _join_key(row, (recording_id_column,))
        if recording_id in recording_ids:
            raise ValueError(f"Duplicate recording ID {recording_id!r}.")
        recording_ids.add(recording_id)
        annotation = Path(rttm_path.format_map(row))
        label = speaker.format_map(row)
        if not label.strip():
            raise ValueError(f"No speaker label configured for recording {recording_id!r}.")
        intervals = _rttm_intervals(annotation, label)
        if not intervals:
            continue

        path = Path(audio_path.format_map(row))
        if not path.is_file():
            raise FileNotFoundError(f"No source recording at {path}.")
        refs = ResourceCollection.from_refs(
            ResourceRef(name=name, kind="text", value=row[column]) for name, column in (resources or {}).items()
        )
        result.extend(
            MetadataSample(utt_id=f"{recording_id}-{index}", path=path, start=start, end=end, resources=refs)
            for index, (start, end) in enumerate(intervals)
        )
    if not result:
        raise ValueError("No RTTM turns matched the configured speaker labels.")
    return result
