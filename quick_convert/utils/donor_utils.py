from pathlib import Path
from omegaconf import OmegaConf


def resolve_donor_paths(h, donor_root: Path, keys: list[str]):
    h = OmegaConf.create(OmegaConf.to_container(h, resolve=True))

    for key in keys:
        if key not in h or h[key] is None:
            continue

        path = Path(h[key])
        if not path.is_absolute():
            h[key] = str(donor_root / path)

    return h
