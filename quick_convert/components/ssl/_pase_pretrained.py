from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

PASE_REVISION = "2a41e63e54fa8673efd12c16cdcdd5ad4f0f125e"
PASEPLUS_CHECKPOINT_ID = "1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW"
PASEPLUS_CONFIG_URL = (
    f"https://raw.githubusercontent.com/santi-pdp/pase/{PASE_REVISION}/cfg/frontend/PASE+.cfg"
)


def _validate_asset(path: Path, *, checkpoint: bool) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"PASE+ download did not produce a nonempty file: {path}")
    if checkpoint:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state)
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise RuntimeError("PASE+ download is not an encoder state dictionary.")
    else:
        with path.open() as file:
            config = json.load(file)
        if not isinstance(config, dict) or not config.get("strides"):
            raise RuntimeError("PASE+ download is not a frontend configuration.")


def _download_asset(destination: Path, *, checkpoint: bool) -> None:
    # A private temporary file plus atomic rename prevents reuse of partial downloads.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temp_dir:
        temporary = Path(temp_dir) / destination.name
        if checkpoint:
            import gdown

            result = gdown.download(id=PASEPLUS_CHECKPOINT_ID, output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError(
                    "Could not download the official PASE+ checkpoint from Google Drive. "
                    "Download FE_e199.ckpt manually and supply both checkpoint_path and config_path."
                )
        else:
            torch.hub.download_url_to_file(PASEPLUS_CONFIG_URL, str(temporary))
        _validate_asset(temporary, checkpoint=checkpoint)
        temporary.replace(destination)


def resolve_paseplus_assets(
    cache_dir: str | Path | None = None, *, local_files_only: bool = False
) -> tuple[Path, Path]:
    """Download/cache the official PASE+ config and encoder checkpoint.

    Uses the PyTorch cache by default (respecting TORCH_HOME), with a directory
    identifying the pinned upstream revision and Google Drive file ID.
    """
    root = Path(cache_dir).expanduser() if cache_dir is not None else Path(torch.hub.get_dir()) / "quick_convert"
    root = root / "paseplus" / f"{PASE_REVISION}-{PASEPLUS_CHECKPOINT_ID}"
    config = root / "PASE+.cfg"
    checkpoint = root / "FE_e199.ckpt"
    paths = ((config, False), (checkpoint, True))
    if local_files_only:
        missing = [str(path) for path, _ in paths if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"PASE+ assets are not cached and local_files_only=True: {', '.join(missing)}")
        return config, checkpoint

    from filelock import FileLock

    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".download.lock")):
        for path, is_checkpoint in paths:
            if not path.is_file() or path.stat().st_size == 0:
                _download_asset(path, checkpoint=is_checkpoint)
    return config, checkpoint
