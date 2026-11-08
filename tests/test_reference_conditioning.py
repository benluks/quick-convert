"""Generic reference pairing, decoder prompts and source-only flow loss."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.features.mel.matcha import MelSpectrogramConfig
from quick_convert.components.generators.acoustic.cosyvoice import CosyVoiceSpectrogramGenerator
from quick_convert.components.generators.acoustic.cosyvoice_reference import CosyVoiceReference, reference_flow_loss
from quick_convert.data.resources import SameSpeakerReferenceProvider


def test_pairing_stays_within_speaker_and_split_and_excludes_source(tmp_path, monkeypatch):
    manifest = tmp_path / "samples.csv"
    manifest.write_text(
        "utt_id,path,split,spkid\n"
        + "".join(
            f"{split}-{speaker}-{i},{split}-{speaker}-{i}.wav,{split},{speaker}\n"
            for split in ("train", "val")
            for speaker in ("one", "two")
            for i in range(3)
        )
    )
    from quick_convert.data import ManifestDataset

    provider = SameSpeakerReferenceProvider(str(manifest))
    other = SameSpeakerReferenceProvider(str(manifest))
    dataset = ManifestDataset(
        manifest,
        resources={"spkid": {"column": "spkid", "kind": "text"}},
        resource_providers=[provider],
        load=["audio"],
    )
    for row in dataset.rows:
        ref = provider(row)
        assert ref.value == other(row).value
        assert str(row.path) != ref.value
        assert row.path.stem.split("-")[:2] == Path(ref.value).stem.split("-")[:2]
        assert "reference_path" not in row.resources.keys()
    monkeypatch.setattr("quick_convert.data.base_dataset.load_audio", lambda *a, **kw: (torch.ones(1, 160), 16000))
    batch = next(iter(dataset.make_dataloader(batch_size=2)))
    assert len(batch.resources["reference_path"]) == 2
    manifest.write_text("utt_id,path,split,spkid\na,a.wav,train,only\n")
    with pytest.raises(ValueError, match="No different"):
        SameSpeakerReferenceProvider(str(manifest))


class ReferenceEncoder(nn.Module):
    def forward(self, features, lengths, streaming=False):
        return features, None


class Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, y, mask, mu, t, spks, cond, streaming=False):
        self.inputs = dict(mask=mask, mu=mu, spks=spks, cond=cond)
        return mu * self.scale + cond


class ReferenceFlow(nn.Module):
    vocab_size = None
    input_size = 2
    output_size = 2
    input_frame_rate = 50
    token_mel_ratio = 1

    def __init__(self):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.encoder_proj = nn.Identity()
        self.spk_embed_affine_layer = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.estimator = Estimator()
        self.decoder.sigma_min = 1e-6
        self.decoder.training_cfg_rate = 0.0

    def output_lengths(self, lengths):
        return lengths

    def inference(self, **kwargs):
        self.inference_inputs = kwargs
        mel = kwargs["token"].transpose(1, 2) + kwargs["prompt_feat"].mean(1)[:, :, None]
        return mel, None


def test_flow_loss_excludes_reference_and_padding_and_compacts_sequences(monkeypatch):
    flow = ReferenceFlow()
    features = torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [999.0, 999.0]]], requires_grad=True)
    lengths = torch.tensor([2, 1])
    reference = CosyVoiceReference(
        torch.tensor([[[10.0, 10.0], [20.0, 20.0]], [[30.0, 30.0], [999.0, 999.0]]]),
        torch.tensor([2, 1]),
        torch.full((2, 2, 2), 1000.0),
        torch.tensor([2, 1]),
    )
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    loss = reference_flow_loss(flow, features, lengths, torch.zeros(2, 2, 2), reference, torch.ones(2, 2))
    torch.testing.assert_close(loss, torch.tensor((1 + 4 + 9) / 3))
    assert flow.decoder.estimator.inputs["mu"][1, 0].tolist() == [30.0, 3.0, 0.0, 0.0]
    assert flow.decoder.estimator.inputs["cond"][1, 0].tolist() == [1000.0, 0.0, 0.0, 0.0]
    loss.backward()
    assert torch.isfinite(flow.decoder.estimator.scale.grad)
    assert features.grad[1, 1].count_nonzero() == 0
    assert features.grad[0].abs().sum() > 0


def decoder(feature_dim=2):
    return CosyVoiceSpectrogramGenerator(
        ReferenceFlow(),
        feature_dim=feature_dim,
        mel_dim=2,
        device="cpu",
        mel_config=MelSpectrogramConfig(16000, 1280, 320, 1280, 2, 0, 8000),
    )


def test_decoder_passes_aligned_prompt_and_preserves_inputs():
    module = decoder()
    features = torch.ones(2, 4, 2)
    lengths = torch.tensor([4, 3])
    reference = CosyVoiceReference(
        torch.ones(2, 2, 2), torch.tensor([2, 2]), torch.full((2, 2, 2), 5.0), torch.tensor([2, 2])
    )
    before = features.clone()
    result = module(features, lengths, torch.ones(2, 2), reference=reference)
    assert module.flow.inference_inputs["prompt_token"].shape == (2, 2, 2)
    assert module.flow.inference_inputs["prompt_feat_len"].tolist() == [2, 2]
    assert result.mel_lengths.tolist() == [4, 3]
    torch.testing.assert_close(result.mel, torch.full((2, 2, 4), 6.0))
    torch.testing.assert_close(features, before)
    with pytest.raises(ValueError, match="aligned"):
        module(features, lengths, torch.ones(2, 2), reference=replace(reference, mel_lengths=torch.tensor([1, 2])))


def test_expanded_reference_does_not_share_storage_with_the_original_or_other_rows():
    reference = CosyVoiceReference(torch.ones(1, 2, 2), torch.tensor([2]), torch.ones(1, 2, 2), torch.tensor([2]))
    expanded = reference.expand(2, device="cpu", dtype=torch.float32)
    expanded.features[0].zero_()
    expanded.mels[0].zero_()
    expanded.lengths[0] = 1
    assert torch.all(reference.features == 1) and torch.all(reference.mels == 1)
    assert torch.all(expanded.features[1] == 1) and torch.all(expanded.mels[1] == 1)
    assert reference.lengths.tolist() == [2]
    assert expanded.lengths.tolist() == [1, 2]


def test_real_cosyvoice_flow_trains_and_generates_with_reference():
    flow_module = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow")
    cfm_module = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow_matching")
    from omegaconf import DictConfig

    from quick_convert.components.sequence.upsample_conformer import UpsampleConformerEncoder

    encoder = UpsampleConformerEncoder(
        input_size=4,
        output_size=4,
        attention_heads=2,
        linear_units=16,
        num_blocks=1,
        use_cnn_module=False,
        macaron_style=False,
        upsample_factor=1,
        post_upsample_blocks=1,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
    )
    cfm = cfm_module.CausalConditionalCFM(
        in_channels=240,
        spk_emb_dim=80,
        estimator=Estimator(),
        cfm_params=DictConfig(
            dict(sigma_min=1e-6, solver="euler", t_scheduler="cosine", training_cfg_rate=0.2, inference_cfg_rate=0.7)
        ),
    )
    flow = flow_module.CausalMaskedDiffWithXvec(
        input_size=4,
        output_size=80,
        spk_embed_dim=2,
        vocab_size=None,
        token_mel_ratio=1,
        encoder=encoder,
        decoder=cfm,
    )
    module = CosyVoiceSpectrogramGenerator(
        flow,
        4,
        device="cpu",
        mel_config=MelSpectrogramConfig(16000, 1280, 320, 1280, 80, 0, 8000),
    )
    features = torch.randn(2, 5, 4, requires_grad=True)
    reference = module.prepare_reference(
        torch.randn(2, 3, 4),
        torch.tensor([3, 2]),
        torch.randn(2, 960),
        torch.tensor([960, 640]),
        16000,
    )
    output = module.compute_loss(
        features,
        torch.tensor([5, 4]),
        torch.randn(2, 1600),
        torch.tensor([1600, 1280]),
        16000,
        torch.randn(2, 2),
        reference=reference,
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert features.grad.abs().sum() > 0
    module.eval()
    prompt = CosyVoiceReference(
        reference.features[:1], reference.lengths[:1], reference.mels[:1], reference.mel_lengths[:1]
    )
    generated = module(
        features.detach(),
        torch.tensor([5, 4]),
        torch.randn(2, 2),
        reference=prompt.expand(2, device="cpu", dtype=torch.float32),
        inference_cfg_rate=0.0,
    )
    assert generated.mel.shape == (2, 80, 5)
    assert generated.mel_lengths.tolist() == [5, 4]


@pytest.mark.parametrize("ratio", [1, 2])
def test_dit_preserves_native_training_and_accepts_native_inference_prompt(monkeypatch, ratio):
    flow_module = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow")
    cfm_module = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow_matching")
    from omegaconf import DictConfig

    from quick_convert.external.cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    cfm = cfm_module.CausalConditionalCFM(
        in_channels=240,
        spk_emb_dim=80,
        estimator=Estimator(),
        cfm_params=DictConfig(
            dict(sigma_min=1e-6, solver="euler", t_scheduler="cosine", training_cfg_rate=0.0, inference_cfg_rate=0.7)
        ),
    )
    flow = flow_module.CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=2,
        vocab_size=None,
        input_frame_rate=50 / ratio,
        token_mel_ratio=ratio,
        pre_lookahead_layer=PreLookaheadLayer(80, 16, 3),
        decoder=cfm,
    )
    module = CosyVoiceSpectrogramGenerator(
        flow,
        feature_dim=4,
        device="cpu",
        mel_config=MelSpectrogramConfig(16000, 1280, 320, 1280, 80, 0, 8000),
    )
    monkeypatch.setattr(flow_module.random, "random", lambda: 0.75)
    monkeypatch.setattr(flow_module.random, "randint", lambda lo, hi: hi)
    features = torch.randn(2, 4, 4, requires_grad=True)
    lengths = torch.tensor([4, 3])
    waveform = torch.randn(2, 1280 * ratio)
    waveform_lengths = torch.tensor([1280, 960]) * ratio
    embedding = torch.randn(2, 2)
    loss = module.compute_loss(features, lengths, waveform, waveform_lengths, 16000, embedding).loss
    mels, _ = module._compute_mels(waveform, waveform_lengths, 16000)
    prefix = int(0.3 * int(lengths[0]) * ratio)
    torch.testing.assert_close(cfm.estimator.inputs["cond"][0, :, :prefix], mels[0, :, :prefix])
    assert cfm.estimator.inputs["cond"][0, :, prefix:].count_nonzero() == 0
    loss.backward()
    assert features.grad.abs().sum() > 0
    reference = module.prepare_reference(
        torch.randn(1, 2, 4),
        torch.tensor([2]),
        torch.randn(1, 640 * ratio),
        torch.tensor([640 * ratio]),
        16000,
    ).expand(2, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="native DiT prefix training"):
        module.compute_loss(features, lengths, waveform, waveform_lengths, 16000, embedding, reference=reference)
    module.eval()
    generated = module(features.detach(), lengths, embedding, reference=reference, inference_cfg_rate=0.3)
    assert generated.mel.shape == (2, 80, 4 * ratio)
    assert generated.mel_lengths.tolist() == [4 * ratio, 3 * ratio]
    torch.testing.assert_close(cfm.estimator.inputs["cond"][:2, :, : 2 * ratio], reference.mels.transpose(1, 2))


@pytest.mark.parametrize("rate", [0.0, 0.3, 0.7])
def test_generation_cfg_override_is_per_call(rate):
    module = decoder()
    args = (torch.ones(1, 4, 2), torch.tensor([4]), torch.ones(1, 2))
    module(*args, inference_cfg_rate=rate)
    assert module.flow.inference_inputs["inference_cfg_rate"] == rate
    module(*args)
    assert "inference_cfg_rate" not in module.flow.inference_inputs


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_generation_rejects_invalid_cfg(rate):
    with pytest.raises(ValueError, match="inference_cfg_rate"):
        decoder()(torch.ones(1, 4, 2), torch.tensor([4]), torch.ones(1, 2), inference_cfg_rate=rate)


def test_cfg_solver_uses_override_without_mutating_default():
    cfm_module = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow_matching")

    class ConstantPrediction(nn.Module):
        def forward(self, x, mask, mu, t, spks, cond, streaming=False):
            # Conditional velocity is 2; unconditional velocity is 1.
            return torch.ones_like(x) + mu

    cfm = cfm_module.ConditionalCFM.__new__(cfm_module.ConditionalCFM)
    nn.Module.__init__(cfm)
    cfm.estimator = ConstantPrediction()
    cfm.inference_cfg_rate = 0.7
    args = dict(
        x=torch.zeros(1, 2, 3),
        t_span=torch.tensor([0.0, 1.0]),
        mu=torch.ones(1, 2, 3),
        mask=torch.ones(1, 1, 3),
        spks=torch.ones(1, 2),
        cond=torch.ones(1, 2, 3),
    )
    for rate in (0.0, 0.3, None):
        result = cfm.solve_euler(**args, inference_cfg_rate=rate)
        torch.testing.assert_close(result, torch.full_like(result, 2 + (0.7 if rate is None else rate)))
        assert cfm.inference_cfg_rate == 0.7
