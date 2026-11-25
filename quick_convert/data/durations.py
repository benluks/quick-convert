"""Metadata-only duration resolution."""
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from quick_convert.utils.audio import probe_audio_duration


def resolve_durations(rows, *, cache_path=None, duration_reader=None):
    reader = duration_reader or probe_audio_duration
    cache_path = Path(cache_path) if cache_path is not None else None
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    durations = {}
    result = []
    for row in rows:
        path = Path(row.path).resolve()
        start = 0.0 if row.start is None else float(row.start)
        if path not in durations:
            stat = path.stat()
            key = str(path)
            signature = [stat.st_size, stat.st_mtime_ns]
            entry = cache.get(key, {})
            if entry.get("signature") == signature:
                duration = float(entry["duration"])
            else:
                duration = float(reader(path))
                cache[key] = {"signature": signature, "duration": duration}
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid audio duration: {path}")
            durations[path] = duration
        end = durations[path] if row.end is None else min(float(row.end), durations[path])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid interval for {row.utt_id}")
        result.append(end - start)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=cache_path.parent, delete=False) as handle:
            json.dump(cache, handle)
            temporary = Path(handle.name)
        temporary.replace(cache_path)
    return result
