from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def resolve_donor_paths(h, donor_root: Path, keys: list[str]):
    if isinstance(h, DictConfig):
        h = OmegaConf.create(OmegaConf.to_container(h, resolve=True))

    for key in keys:
        if not hasattr(h, key):
            continue

        value = getattr(h, key)
        if value is None:
            continue

        path = Path(value)
        if not path.is_absolute():
            setattr(h, key, str(donor_root / path))

    return h
