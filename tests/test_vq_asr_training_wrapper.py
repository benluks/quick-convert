from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.layers.heads import HeadOutput
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch


pytest.importorskip("lightning")

from quick_convert.pipelines.training.modules.vq_asr import VQASRTrainingModule  # noqa: E402
from quick_convert.pipelines.training.optim.base import Optimization  # noqa: E402


class FakeQuantizer(nn.Module):
    n_codebooks = 1
    codebook_size = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    @staticmethod
    def output_lengths(lengths):
        return lengths

    def forward(self, features, padding_mask):
        self.calls += 1
        z_q = features * self.scale
        loss = z_q.mean() * 0 + 0.25
        return RVQOutput(
            z_q=z_q,
            layer_z_qs=[z_q],
            codes=torch.zeros(features.shape[0], 1, features.shape[2], dtype=torch.long),
            latents=z_q + 1,
            loss=RVQLosses(loss=loss, raw={"commitment": loss}, weighted={"commitment": loss}),
        )


class FakeCTCHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.predict_calls = 0
        self.loss_calls = 0

    def predict(self, features, *, lengths, padding_mask):
        self.predict_calls += 1
        logits = features + self.bias
        return HeadOutput(predictions=logits, states={"logits": logits})

    def compute_loss_from_logits(self, logits, *, targets, lengths):
        self.loss_calls += 1
        assert lengths.tolist() == [3, 2]
        assert targets.lengths.tolist() == [2, 1]
        loss = logits.mean() * 0 + 0.5
        return HeadOutput(loss=loss, predictions=logits, states={"logits": logits})


class FakeOnlineEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))


def make_batch():
    return AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={
            "content": ResolvedResource(
                values=torch.ones(2, 3, 4),
                lengths=torch.tensor([3, 2]),
            ),
            "token_ids": ResolvedResource(
                values=torch.tensor([[1, 2], [3, 0]]),
                lengths=torch.tensor([2, 1]),
            ),
        },
    )


def make_module(**kwargs):
    return VQASRTrainingModule(
        quantizer=FakeQuantizer(),
        ctc_head=FakeCTCHead(),
        optimization=Optimization(lr_scheduler=None),
        **kwargs,
    )


def test_training_step_reuses_the_system_forward_pass(monkeypatch):
    module = make_module()
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)

    output = module._shared_step(make_batch(), "val")

    assert module.quantizer.calls == 1
    assert module.ctc_head.predict_calls == 1
    assert module.ctc_head.loss_calls == 1
    torch.testing.assert_close(output.loss, torch.tensor(0.75))
    assert output.vq is not None
    assert output.ctc.states["logits"].shape == (2, 3, 4)


def test_legacy_checkpoint_keys_are_mapped_to_the_wrapped_system():
    module = make_module()
    current_state = module.state_dict()
    legacy_state = {key.removeprefix("system."): value.clone() for key, value in current_state.items()}

    prepared_state = module._prepare_checkpoint_state_dict(legacy_state)

    assert prepared_state.keys() == current_state.keys()
    for key, value in current_state.items():
        torch.testing.assert_close(prepared_state[key], value)
    module.load_state_dict(prepared_state, strict=True)


def test_new_checkpoint_keys_remain_unchanged():
    module = make_module()
    current_state = module.state_dict()

    prepared_state = module._prepare_checkpoint_state_dict(dict(current_state))

    assert prepared_state.keys() == current_state.keys()
    assert all(prepared_state[key] is value for key, value in current_state.items())


def test_online_encoders_remain_excluded_from_checkpoints_by_default():
    module = make_module(online_encoders={"content": FakeOnlineEncoder()})
    checkpoint = {"state_dict": dict(module.state_dict())}

    module.on_save_checkpoint(checkpoint)

    assert "system.quantizer.scale" in checkpoint["state_dict"]
    assert not any(key.startswith("system.online_encoders.") for key in checkpoint["state_dict"])
