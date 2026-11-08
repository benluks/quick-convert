"""Independent stream visibility, target-only gradients, and compact prefix inference."""

from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.features.mel.matcha import MelSpectrogramConfig
from quick_convert.components.generators.acoustic import (
    ConditioningStream,
    CosyVoiceSpectrogramGenerator,
    StreamConditioning,
)
from quick_convert.components.generators.acoustic.stream_flow import stream_flow_loss


def test_shared_target_and_independent_context_masks_preserve_inputs():
    torch.manual_seed(42)
    conditioning = StreamConditioning(
        {"speaker": 2, "prosody": 1},
        prefix_fraction=(0.5, 0.5),
        context_mask_fraction=(0.25, 0.25),
        stream_drop_probability=0,
    )
    lengths = torch.tensor([12, 8])
    values = torch.randn(2, 12, 2, requires_grad=True)
    original = values.detach().clone()
    packed, target = conditioning.training_conditions(
        {"speaker": ConditioningStream(values, lengths), "prosody": ConditioningStream(values[..., :1], lengths)},
        lengths,
        values,
    )
    assert target.sum(1).tolist() == [6, 4]
    assert not packed[..., 2][target].any()
    assert not packed[..., 4][target].any()
    assert packed[1, 8:].count_nonzero() == 0
    packed.sum().backward()
    assert values.grad[target].count_nonzero() == 0
    assert values.grad[1, 8:].count_nonzero() == 0
    torch.testing.assert_close(values, original)


def test_prefix_lengths_missing_streams_and_padding():
    conditioning = StreamConditioning({"a": 1, "b": 2})
    values = torch.tensor([[[2.0], [3.0], [float("nan")]], [[4.0], [float("nan")], [float("nan")]]])
    packed, lengths = conditioning.prefix_conditions(
        {"a": ConditioningStream(values, torch.tensor([2, 1]))}, torch.zeros(2, 4, 1)
    )
    assert lengths.tolist() == [2, 1]
    assert packed.shape == (2, 2, 5)
    assert torch.isfinite(packed).all()
    assert packed[..., 2:].count_nonzero() == 0
    assert packed[1, 1].count_nonzero() == 0
    packed, lengths = conditioning.prefix_conditions({}, torch.zeros(2, 4, 1))
    assert packed.shape == (2, 0, 5)
    assert lengths.tolist() == [0, 0]


class Estimator(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, y, mask, mu, t, spks, cond, streaming=False):
        self.calls.append((mu.detach().clone(), cond.detach().clone(), mask.clone()))
        return mu * self.scale + cond[:, :1].expand_as(y)


class Flow(nn.Module):
    vocab_size = None
    input_size = 2
    output_size = 2
    input_frame_rate = 50
    token_mel_ratio = 1

    def __init__(self, cond_dim=3):
        super().__init__()
        self.pre_lookahead_layer = nn.Identity()
        self.spk_embed_affine_layer = nn.Identity()
        self.decoder = nn.Module()
        self.decoder.estimator = Estimator(cond_dim)
        self.decoder.sigma_min = 0.0
        self.decoder.training_cfg_rate = 0.0
        self.decoder.inference_cfg_rate = 0.0
        self.decoder.t_scheduler = "linear"

    def output_lengths(self, lengths):
        return lengths * self.token_mel_ratio


def decoder():
    return CosyVoiceSpectrogramGenerator(
        Flow(),
        feature_dim=2,
        mel_dim=2,
        device="cpu",
        mel_config=MelSpectrogramConfig(
            sample_rate=16000, hop_size=320, num_mels=2, n_fft=1024, win_size=1024, fmin=0, fmax=None
        ),
        stream_conditioning=StreamConditioning({"prosody": 2}, stream_drop_probability=0),
        stream_inference_steps=2,
    )


def test_loss_has_no_direct_gradient_outside_shared_target(monkeypatch):
    flow = Flow()
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    h = torch.ones(2, 4, 2, requires_grad=True)
    target = torch.zeros(2, 4, 2, requires_grad=True)
    conditions = torch.zeros(2, 4, 3)
    score = torch.tensor([[False, True, False, False], [True, False, False, False]])
    loss = stream_flow_loss(flow, h, torch.ones(2, 2), target, torch.tensor([4, 2]), conditions, score)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert target.grad[~score].count_nonzero() == 0
    assert h.grad[~score].count_nonzero() == 0
    assert h.grad[score].count_nonzero() > 0


def test_decoder_training_and_prefix_only_inference(monkeypatch):
    model = decoder()
    lengths = torch.tensor([4, 2])
    features = torch.randn(2, 4, 2, requires_grad=True)
    values = torch.randn(2, 4, 2)
    original = values.clone()
    monkeypatch.setattr(model, "_compute_mels", lambda *a: (torch.zeros(2, 2, 4), lengths))
    loss = model.compute_loss(
        features,
        lengths,
        torch.zeros(2, 1280),
        torch.tensor([1280, 640]),
        16000,
        torch.ones(2, 2),
        streams={"prosody": ConditioningStream(values, lengths)},
    ).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.flow.decoder.estimator.scale.grad is not None
    model.flow.decoder.estimator.calls.clear()
    prefixes = {"prosody": ConditioningStream(values, torch.tensor([3, 1]))}
    result = model(features.detach(), lengths, torch.ones(2, 2), streams=prefixes, inference_cfg_rate=0.5)
    assert result.mel.shape == (2, 2, 4)
    assert result.mel_lengths.tolist() == [4, 2]
    assert result.mel[1, :, 2:].count_nonzero() == 0
    mu, cond, mask = model.flow.decoder.estimator.calls[0]
    assert mu.shape[-1] == 7
    assert cond[0, :, 3:].count_nonzero() == 0
    assert cond[1, :, 1:].count_nonzero() == 0
    assert mask[1].sum() == 3
    # Guidance preserves linguistic conditioning in the null auxiliary branch.
    torch.testing.assert_close(model.flow.decoder.estimator.calls[1][0], mu)
    assert model.flow.decoder.estimator.calls[1][1].count_nonzero() == 0
    torch.testing.assert_close(values, original)
    assert lengths.tolist() == [4, 2]
    assert model(features.detach(), lengths, torch.ones(2, 2)).mel.shape == (2, 2, 4)


def test_bad_streams_fail_explicitly():
    model = decoder()
    with pytest.raises(ValueError, match="Undeclared"):
        model(
            torch.zeros(1, 4, 2),
            torch.tensor([4]),
            torch.ones(1, 2),
            streams={"typo": ConditioningStream(torch.zeros(1, 2, 2), torch.tensor([2]))},
        )
    conditioning = model.stream_conditioning
    with pytest.raises(ValueError, match="cover each target"):
        conditioning.training_conditions(
            {"prosody": ConditioningStream(torch.zeros(1, 4, 2), torch.tensor([3]))},
            torch.tensor([4]),
            torch.zeros(1, 4, 2),
        )
    with pytest.raises(ValueError, match="cond_dim"):
        CosyVoiceSpectrogramGenerator(Flow(2), 2, mel_dim=2, stream_conditioning=conditioning)


def test_stream_run_composes_without_loading_models():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name="run/train_sslr_w2vbert_dit_streams")
        conditioning = instantiate(cfg.system.decoder.stream_conditioning)
        assert conditioning.output_dim == cfg.system.decoder.flow.decoder.estimator.cond_dim
        assert conditioning.dimensions == {"mel": 80}


def test_stream_run_names_are_distinct_from_each_other_and_the_base_run():
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    names = []
    previous = HydraConfig.instance().cfg
    try:
        with initialize_config_dir(
            version_base=None, config_dir=str(Path(__file__).parents[1] / "quick_convert/configs")
        ):
            for run in (
                "train_sslr_w2vbert_dit",
                "train_sslr_w2vbert_dit_streams",
                "train_sslr_w2vbert_dit_streams_no_xvector",
                "train_sslr_w2vbert_dit_campp_prefix",
                "train_sslr_dac_dit_campp_prefix",
            ):
                config = compose(config_name="run/" + run, return_hydra_config=True)
                HydraConfig.instance().set_config(config)
                names.append(config.exp_name)
    finally:
        HydraConfig.instance().cfg = previous
    assert len(set(names)) == len(names)
    assert all("/" not in name and "\\" not in name for name in names)


def test_real_dit_supports_non_mel_condition_width():
    DiT = pytest.importorskip("quick_convert.external.cosyvoice.flow.DiT.dit").DiT

    model = DiT(dim=32, depth=1, heads=2, dim_head=16, mel_dim=2, mu_dim=2, spk_dim=2, cond_dim=5)
    x = torch.randn(2, 2, 6, requires_grad=True)
    output = model(
        x,
        torch.ones(2, 1, 6, dtype=torch.bool),
        torch.randn(2, 2, 6),
        torch.rand(2),
        torch.ones(2, 2),
        torch.randn(2, 5, 6),
    )
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_checkpoint_rejects_reordered_equal_width_streams():
    state = StreamConditioning({"speaker": 2, "prosody": 2}).state_dict()
    with pytest.raises(ValueError, match="names, order, or dimensions"):
        StreamConditioning({"prosody": 2, "speaker": 2}).load_state_dict(state)
    StreamConditioning({"speaker": 2, "prosody": 2}).load_state_dict(state)


def test_actual_dit_stream_training_and_generation_with_upsampling(monkeypatch):
    DiT = pytest.importorskip("quick_convert.external.cosyvoice.flow.DiT.dit").DiT

    conditioning = StreamConditioning({"mel": 2, "pitch": 1}, stream_drop_probability=0)
    flow = Flow(cond_dim=5)
    flow.token_mel_ratio = 2
    flow.input_frame_rate = 25
    flow.decoder.estimator = DiT(dim=32, depth=1, heads=2, dim_head=16, mel_dim=2, mu_dim=2, spk_dim=2, cond_dim=5)
    model = CosyVoiceSpectrogramGenerator(
        flow,
        feature_dim=2,
        mel_dim=2,
        device="cpu",
        mel_config=MelSpectrogramConfig(
            sample_rate=16000, hop_size=320, num_mels=2, n_fft=1024, win_size=1024, fmin=0, fmax=None
        ),
        stream_conditioning=conditioning,
        stream_inference_steps=1,
    )
    features = torch.randn(2, 3, 2, requires_grad=True)
    lengths = torch.tensor([3, 2])
    mel_lengths = lengths * 2
    monkeypatch.setattr(model, "_compute_mels", lambda *a: (torch.randn(2, 2, 6), mel_lengths))
    output = model.compute_loss(
        features,
        lengths,
        torch.zeros(2, 1920),
        mel_lengths * 320,
        16000,
        torch.ones(2, 2),
        streams={"pitch": ConditioningStream(torch.randn(2, 6, 1), mel_lengths)},
    )
    output.loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[0].abs().sum() > 0
    generated = model(
        features.detach(),
        lengths,
        torch.ones(2, 2),
        streams={
            "pitch": ConditioningStream(torch.randn(2, 3, 1), torch.tensor([3, 1])),
            "mel": ConditioningStream(torch.randn(2, 2, 2), torch.tensor([1, 2])),
        },
    )
    assert generated.mel.shape == (2, 2, 6)
    assert generated.mel_lengths.tolist() == [6, 4]
    assert torch.isfinite(generated.mel).all()
    assert generated.mel[1, :, 4:].count_nonzero() == 0


def test_system_resolves_training_resources_but_uses_explicit_inference_prefixes():
    from quick_convert.components.resources.resolution import ResolvedResource
    from quick_convert.data import AudioBatch
    from quick_convert.systems.conversion import SSLReconstructionConversionSystem

    model = decoder()
    system = SSLReconstructionConversionSystem(model)
    lengths = torch.tensor([4])
    batch = AudioBatch(
        utt_ids=["a"],
        paths=[Path("a.wav")],
        splits=[None],
        resources={
            "content": ResolvedResource(torch.ones(1, 4, 2), lengths),
            "speaker": ResolvedResource(torch.ones(1, 2)),
            "prosody": ResolvedResource(torch.full((1, 4, 2), 9.0), lengths),
        },
    )
    streams = system.conditioning_streams(batch)
    assert streams["prosody"].lengths.tolist() == [4]
    prefix = ConditioningStream(torch.ones(1, 2, 2), torch.tensor([2]))
    result = system(batch, streams={"prosody": prefix}, run_vocoder=False)
    assert result.generation.mel_lengths.tolist() == [4]
    _, cond, _ = model.flow.decoder.estimator.calls[0]
    assert cond[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_no_xvector_stream_model_trains_and_generates_without_speaker_resource(monkeypatch):
    pytest.importorskip("lightning")
    pytest.importorskip("quick_convert.external.cosyvoice.flow.DiT.dit")
    from quick_convert.components.resources.resolution import ResolvedResource
    from quick_convert.data import AudioBatch
    from quick_convert.external.cosyvoice.flow.DiT.dit import DiT
    from quick_convert.systems.conversion import SSLReconstructionConversionSystem
    from quick_convert.training.lightning.modules.ssl_reconstruction import SSLReconstructionTrainingModule
    from quick_convert.training.lightning.optim import Optimization

    flow = Flow()
    flow.decoder.estimator = DiT(
        dim=32,
        depth=1,
        heads=2,
        dim_head=16,
        mel_dim=2,
        mu_dim=2,
        spk_dim=None,
        cond_dim=3,
    )
    model = CosyVoiceSpectrogramGenerator(
        flow,
        feature_dim=2,
        mel_dim=2,
        device="cpu",
        mel_config=MelSpectrogramConfig(16000, 1024, 320, 1024, 2, 0, None),
        stream_conditioning=StreamConditioning({"mel": 2}),
        stream_inference_steps=1,
    )
    assert not model.requires_speaker_embedding
    assert flow.spk_embed_affine_layer is None
    assert not any("spk_embed_affine_layer" in key for key in model.state_dict())
    lengths = torch.tensor([4, 2])
    monkeypatch.setattr(model, "_compute_mels", lambda *a, **kw: (torch.zeros(2, 2, 4), lengths))
    system = SSLReconstructionConversionSystem(model, online_encoders={"speaker": None})
    assert "speaker" not in system.online_encoders
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={"content": ResolvedResource(torch.randn(2, 4, 2), lengths)},
        waveforms=torch.zeros(2, 1280),
        lengths=lengths * 320,
        sample_rates=torch.tensor([16000, 16000]),
    )
    training = SSLReconstructionTrainingModule(system=system, optimization=Optimization(lr_scheduler=None))
    monkeypatch.setattr(training, "log_dict", lambda *a, **kw: None)
    output = training._shared_step(batch, "val")
    output.loss.backward()
    assert output.speaker_embedding is None
    assert torch.isfinite(output.loss)
    assert model.flow.decoder.estimator.input_embed.proj.weight.grad is not None
    result = system(batch, run_vocoder=False)
    assert result.speaker_embedding is None
    assert result.generation.mel_lengths.tolist() == [4, 2]

    class ContentEncoder(nn.Module):
        sample_rate = 16000
        device = torch.device("cpu")

        def forward(self, batch):
            return ResolvedResource(torch.ones(len(batch), 4, 2), torch.full((len(batch),), 4))

    system.online_encoders["content"] = ContentEncoder()
    reconstructed = system.reconstruct(torch.zeros(1280), sample_rate=16000, run_vocoder=False)
    assert reconstructed.speaker_embedding is None
    assert reconstructed.generation.mel_lengths.tolist() == [4]
    # Exercise both guidance branches with no speaker tensors.
    result = model(torch.randn(2, 4, 2), lengths, inference_cfg_rate=1.0)
    assert torch.isfinite(result.mel).all()
    with pytest.raises(ValueError, match="omit speaker_embedding"):
        model(torch.zeros(2, 4, 2), lengths, speaker_embedding=torch.ones(2, 2))
    with pytest.raises(ValueError, match="requires a speaker_embedding"):
        decoder()(torch.zeros(2, 4, 2), lengths)


def test_no_xvector_run_disables_speaker_encoder_and_dit_input():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name="run/train_sslr_w2vbert_dit_streams_no_xvector")
        assert cfg.system.online_encoders.speaker is None
        assert cfg.system.decoder.flow.decoder.estimator.spk_dim is None


def test_prefixes_are_left_padded_per_sample_and_end_at_generation_boundary():
    conditioning = StreamConditioning({"a": 1, "b": 1})
    a = torch.tensor([[[1.0], [2.0], [999.0]], [[3.0], [999.0], [999.0]]], requires_grad=True)
    b = torch.tensor([[[4.0], [5.0], [6.0]], [[7.0], [8.0], [999.0]]])
    packed, lengths = conditioning.prefix_conditions(
        {
            "a": ConditioningStream(a, torch.tensor([2, 1])),
            "b": ConditioningStream(b, torch.tensor([3, 2])),
        },
        torch.zeros(2, 5, 1),
    )
    assert lengths.tolist() == [3, 2]
    assert packed[0, :, 0].tolist() == [0, 1, 2]
    assert packed[0, :, 1].tolist() == [0, 1, 1]
    assert packed[1, :, 0].tolist() == [0, 3, 0]
    assert packed[1, :, 2].tolist() == [7, 8, 0]
    packed.sum().backward()
    assert a.grad[0, :, 0].tolist() == [1, 1, 0]
    assert a.grad[1, :, 0].tolist() == [1, 0, 0]


def test_training_samples_independent_prefixes_and_scores_only_the_continuation():
    torch.manual_seed(9)
    conditioning = StreamConditioning(
        {"a": 1, "b": 1}, prefix_fraction=(0.1, 0.6), context_mask_fraction=(0, 0), stream_drop_probability=0
    )
    lengths = torch.tensor([100, 80])
    values = torch.arange(100.0).view(1, 100, 1).expand(2, -1, -1)
    packed, score = conditioning.training_conditions(
        {
            "a": ConditioningStream(values, lengths),
            "b": ConditioningStream(values, lengths),
        },
        lengths,
        values,
    )
    a_lengths = packed[..., 1].sum(1).long()
    b_lengths = packed[..., 3].sum(1).long()
    assert not torch.equal(a_lengths, b_lengths)
    for i, length in enumerate(lengths.tolist()):
        boundary = int(torch.maximum(a_lengths[i], b_lengths[i]))
        assert not score[i, :boundary].any()
        assert score[i, boundary:length].all()
        assert not score[i, length:].any()
        assert packed[i, boundary:].count_nonzero() == 0
        for offset, sizes in ((0, a_lengths), (2, b_lengths)):
            size = int(sizes[i])
            assert packed[i, : boundary - size, offset : offset + 2].count_nonzero() == 0
            torch.testing.assert_close(packed[i, boundary - size : boundary, offset], torch.arange(float(size)))


def test_all_streams_dropped_score_all_valid_frames_and_single_frame_is_valid():
    conditioning = StreamConditioning({"a": 1}, prefix_fraction=(1, 1), stream_drop_probability=1)
    lengths = torch.tensor([4, 1])
    packed, score = conditioning.training_conditions(
        {"a": ConditioningStream(torch.ones(2, 4, 1), lengths)}, lengths, torch.ones(2, 4, 1)
    )
    assert packed.count_nonzero() == 0
    assert score.sum(1).tolist() == [4, 1]
    conditioning.stream_drop_probability = 0
    _, score = conditioning.training_conditions(
        {"a": ConditioningStream(torch.ones(2, 4, 1), lengths)}, lengths, torch.ones(2, 4, 1)
    )
    assert score.sum(1).tolist() == [1, 1]


def test_float32_frozen_prefixes_work_during_autocast_generation():
    model = decoder()
    model.flow.pre_lookahead_layer = nn.Linear(2, 2)
    values = torch.randn(1, 2, 2)
    original = values.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(
            torch.randn(1, 4, 2),
            torch.tensor([4]),
            torch.ones(1, 2),
            streams={"prosody": ConditioningStream(values, torch.tensor([2]))},
        )
    assert torch.isfinite(result.mel).all()
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, original)


def test_named_reference_conversion_keeps_reference_content_in_both_guidance_branches(tmp_path, monkeypatch):
    from quick_convert.components.resources.resolution import ResolvedResource
    from quick_convert.data import AudioBatch
    from quick_convert.systems.conversion import ReferenceSpeech, SSLReconstructionConversionSystem

    class Content(nn.Module):
        sample_rate = 16000

        def forward(self, batch):
            lengths = batch.lengths // 320
            values = batch.waveforms[:, :1, None].expand(-1, int(lengths.max()), 2).clone()
            return ResolvedResource(values, lengths)

    class Auxiliary(nn.Module):
        sample_rate = 16000

        def encode_prefixes(self, batch, lengths, frame_rate):
            return ResolvedResource(torch.full((len(batch), int(lengths.max()), 2), 9.0), lengths)

    class Vocoder(nn.Module):
        samples_per_frame = 320
        sample_rate = 16000

        def forward(self, mel):
            return mel[:, 0].repeat_interleave(self.samples_per_frame, dim=1)

    model = decoder()
    monkeypatch.setattr(model, "_get_vocoder", lambda: Vocoder())
    system = SSLReconstructionConversionSystem(
        model,
        online_encoders={"content": Content(), "speaker": nn.Identity(), "prosody": Auxiliary()},
    ).eval()
    # The speaker encoder is model-specific; provide a stable global vector
    # while exercising real reference and source content extraction.
    monkeypatch.setattr(system, "speaker_condition", lambda batch: torch.ones(len(batch), 2))
    monkeypatch.setattr(
        "quick_convert.systems.conversion.ssl_reconstruction.load_audio_input",
        lambda *args, **kwargs: torch.full((1, 640), 7.0),
    )
    target = system.prepare_target(ReferenceSpeech([tmp_path / "ref.wav"], prompt_seconds=0.04))
    source = AudioBatch(
        utt_ids=["a"],
        paths=[tmp_path / "source.wav"],
        splits=[None],
        resources={},
        waveforms=torch.ones(1, 1280),
        lengths=torch.tensor([1280]),
        sample_rates=torch.tensor([16000]),
    )

    audio = system.convert_batch(source, target=target, inference_cfg_rate=0.5)
    assert audio.lengths.tolist() == [1280]
    conditioned, unconditioned = model.flow.decoder.estimator.calls[:2]
    expected = torch.tensor([[[7, 7, 1, 1, 1, 1], [7, 7, 1, 1, 1, 1]]], dtype=torch.float32)
    torch.testing.assert_close(conditioned[0], expected)
    torch.testing.assert_close(unconditioned[0], expected)
    assert conditioned[1][..., :2].count_nonzero() > 0
    assert unconditioned[1].count_nonzero() == 0
