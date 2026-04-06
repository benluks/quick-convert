from pathlib import Path


def parent_dir(wav_path: Path) -> str:
    return wav_path.parent.name


def grandparent_dir(wav_path: Path) -> str:
    return wav_path.parent.parent.name


def stem_prefix(wav_path: Path) -> str:
    return wav_path.stem.split("_")[0]


def stem_suffix(wav_path: Path) -> str:
    return wav_path.stem.split("_")[-1]
