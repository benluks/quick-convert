import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quick_convert.components.ssl import PASEContentEncoder
from quick_convert.components.ssl._pase_pretrained import (
    PASEPLUS_CHECKPOINT_ID,
    PASEPLUS_CONFIG_URL,
    resolve_paseplus_assets,
)


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def config(url, output):
        calls.append(("config", url))
        Path(output).write_text('{"strides": [1, 10, 2, 1, 2, 1, 2, 2]}')

    def checkpoint(*, id, output, quiet):
        calls.append(("checkpoint", id))
        torch.save({"state_dict": {"weight": torch.ones(1)}}, output)
        return output

    monkeypatch.setattr(torch.hub, "download_url_to_file", config)
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=checkpoint))
    return calls


def test_paseplus_downloads_once_and_reuses_offline(tmp_path, downloads):
    config, checkpoint = resolve_paseplus_assets(tmp_path)
    assert downloads == [("config", PASEPLUS_CONFIG_URL), ("checkpoint", PASEPLUS_CHECKPOINT_ID)]
    assert json.loads(config.read_text())["strides"][1] == 10
    assert torch.load(checkpoint, weights_only=True)["state_dict"]["weight"].item() == 1
    assert resolve_paseplus_assets(tmp_path) == (config, checkpoint)
    assert resolve_paseplus_assets(tmp_path, local_files_only=True) == (config, checkpoint)
    assert len(downloads) == 2


def test_paseplus_offline_miss_never_downloads(tmp_path, downloads):
    with pytest.raises(FileNotFoundError, match="local_files_only"):
        resolve_paseplus_assets(tmp_path, local_files_only=True)
    assert not downloads
    assert not list(tmp_path.iterdir())


def test_paseplus_default_uses_torch_cache(tmp_path, monkeypatch, downloads):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    config, _ = resolve_paseplus_assets()
    assert config.is_relative_to(tmp_path / "quick_convert" / "paseplus")


@pytest.mark.parametrize("failure", ["exception", "empty", "html", "none"])
def test_paseplus_failed_download_is_not_cached_and_can_retry(tmp_path, monkeypatch, downloads, failure):
    original = sys.modules["gdown"].download

    def broken(**kwargs):
        path = Path(kwargs["output"])
        if failure == "exception":
            path.write_bytes(b"partial")
            raise OSError("Interrupted")
        if failure == "html":
            path.write_text("<html>Google Drive quota exceeded</html>")
        if failure == "empty":
            path.touch()
        return None if failure == "none" else str(path)

    monkeypatch.setattr(sys.modules["gdown"], "download", broken)
    with pytest.raises((OSError, RuntimeError, ValueError, EOFError, KeyError, IndexError, pickle.UnpicklingError)):
        resolve_paseplus_assets(tmp_path)
    assert not list(tmp_path.rglob("FE_e199.ckpt"))
    assert not list(tmp_path.rglob(".download-*"))
    monkeypatch.setattr(sys.modules["gdown"], "download", original)
    config, checkpoint = resolve_paseplus_assets(tmp_path)
    assert config.is_file() and checkpoint.is_file()


def test_paseplus_requires_a_complete_custom_pair():
    with pytest.raises(ValueError, match="both"):
        PASEContentEncoder(config_path="custom.cfg", device="cpu")
    with pytest.raises(ValueError, match="both"):
        PASEContentEncoder(checkpoint_path="custom.ckpt", device="cpu")


def test_paseplus_default_assets_reach_strict_model_loading(tmp_path, monkeypatch, downloads):
    model = torch.nn.Linear(1, 1, bias=False)
    monkeypatch.setitem(sys.modules, "pase", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models.frontend", SimpleNamespace(wf_builder=lambda config: model))
    # Downloaded keys deliberately mismatch: loading must fail rather than keep random weights.
    with pytest.raises(RuntimeError, match="size mismatch"):
        PASEContentEncoder(cache_dir=tmp_path, device="cpu")


@pytest.mark.skipif(os.environ.get("QC_PASE_DOWNLOAD_TEST") != "1", reason="Downloads official PASE+ assets")
def test_paseplus_published_download_and_offline_reuse(tmp_path):
    config, checkpoint = resolve_paseplus_assets(tmp_path)
    assert json.loads(config.read_text())["denseskips"] is True
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    assert len(state) > 20
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert resolve_paseplus_assets(tmp_path, local_files_only=True) == (config, checkpoint)
