from types import SimpleNamespace

import pytest
import torch
from torch import nn


pytest.importorskip("lightning")

from quick_convert.training.lightning.trainer import LightningTrainer


def test_compile_replaces_modules_on_their_actual_owner(monkeypatch):
    decoder = nn.Linear(2, 1)
    direct = nn.Linear(2, 1)
    module = nn.Module()
    module.system = nn.Module()
    module.system.decoder = decoder
    module.direct = direct
    module.label = "metadata"
    replacements = {decoder: nn.Identity(), direct: nn.Identity()}
    calls = []

    def compile(value, **kwargs):
        calls.append((value, kwargs))
        return replacements[value]

    monkeypatch.setattr(torch, "compile", compile)
    trainer = SimpleNamespace(
        module=module,
        compile_cfg={"enabled": True, "targets": ["system.decoder", "direct", "missing", "label"]},
    )
    LightningTrainer._maybe_compile_module(trainer)
    assert module.system.decoder is replacements[decoder]
    assert module.direct is replacements[direct]
    assert [value for value, _ in calls] == [decoder, direct]
    assert module.label == "metadata"
