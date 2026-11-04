"""Paired variable-duration DiT reconstruction without model downloads."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from test_stream_conditioning import Flow

from quick_convert.components.decoders import ConditioningStream, CosyVoiceSpectrogramGenerator, StreamConditioning
from quick_convert.components.encoders.upsample_conformer import UpsampleConformerEncoder
from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig
from quick_convert.external.cosyvoice.flow.DiT.dit import DiT
from quick_convert.systems.reconstruction import SSLReconstructionSystem


def decoder():
    flow = Flow()
    flow.decoder.estimator = DiT(dim=32, depth=1, heads=2, dim_head=16, mel_dim=2, mu_dim=2, spk_dim=None, cond_dim=3)
    return CosyVoiceSpectrogramGenerator(
        flow,
        feature_dim=4,
        mel_dim=2,
        device="cpu",
        content_encoder=UpsampleConformerEncoder(
            input_size=4,
            output_size=8,
            attention_heads=2,
            linear_units=16,
            num_blocks=1,
            use_cnn_module=False,
            dropout_rate=0,
            positional_dropout_rate=0,
        ),
        mel_config=MelSpectrogramConfig(16000, 1024, 320, 1024, 2, 0, None),
        stream_conditioning=StreamConditioning({"speaker": 2}, context_mask_fraction=(0, 0), stream_drop_probability=0),
        stream_inference_steps=1,
    )


def test_real_dit_paired_loss_generation_and_checkpoint(monkeypatch):
    model = decoder()
    features = torch.randn(2, 6, 4, requires_grad=True)
    lengths = torch.tensor([6, 3])
    target_lengths = torch.tensor([4, 8])
    prefix = ConditioningStream(torch.randn(2, 5, 2), torch.tensor([5, 2]), is_prefix=True)
    targets = torch.randn(2, 2, 8)
    monkeypatch.setattr(model, "_compute_mels", lambda *a: (targets, target_lengths))
    output = model.compute_loss(
        features, lengths, torch.zeros(2, 2560), target_lengths * 320, 16000, streams={"speaker": prefix}
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    assert features.grad[1, 3:].count_nonzero() == 0
    assert model.flow.pre_lookahead_layer is None
    clone = decoder()
    clone.load_state_dict(model.state_dict(), strict=True)
    system = SSLReconstructionSystem(clone).eval()
    result = system.generate_features(
        features.detach(),
        lengths,
        duration_ratio=target_lengths.double() / lengths,
        streams={"speaker": prefix},
        run_vocoder=False,
    )
    assert result.generation.mel_lengths.tolist() == [4, 8]
    assert result.generation.mel.shape == (2, 2, 8)
    assert result.generation.mel[0, :, 4:].count_nonzero() == 0
    assert torch.isfinite(result.generation.mel).all()


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf"), [1, 2, 3]])
def test_invalid_duration_ratio(ratio):
    with pytest.raises(ValueError, match="duration_ratio"):
        decoder().requested_mel_lengths(torch.tensor([3, 5]), ratio)


def test_independent_prefix_scores_every_target_frame(monkeypatch):
    import quick_convert.components.decoders.stream_flow as module

    def loss(flow, h, spks, targets, lengths, conditions, score):
        assert lengths.tolist() == [7, 8]
        assert score.sum(1).tolist() == [2, 6]
        assert h[0, :5].count_nonzero() == 0
        torch.testing.assert_close(targets[0, 5:7], torch.ones(2, 2))
        return targets.sum()

    monkeypatch.setattr(module, "stream_flow_loss", loss)
    module.independent_prefix_loss(
        None,
        StreamConditioning({"speaker": 2}, context_mask_fraction=(0, 0), stream_drop_probability=0),
        torch.ones(2, 6, 2),
        None,
        torch.ones(2, 6, 2),
        torch.tensor([2, 6]),
        {"speaker": ConditioningStream(torch.ones(2, 5, 2), torch.tensor([5, 2]), is_prefix=True)},
    )


def test_paired_run_routes_content_through_tts_audio_resource():
    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(
            config_name="run/train_sslr_tts_dit_campp_prefix",
            overrides=["tts_data_path=/tmp/tts", "dataset_path=/tmp/original"],
        )
        assert cfg.system.online_encoders.content.layer == -1
        assert cfg.system.feature_transform is None
        assert cfg.system.decoder.flow.pre_lookahead_layer is None
        assert cfg.train_dataset.load == ["audio"]
        assert cfg.val_dataset.target_sr == 16000
        encoder = instantiate(cfg.system.decoder.content_encoder)
        assert encoder.output_size() == 256
        assert cfg.system.decoder.flow.decoder.estimator.cond_dim == 513
        from quick_convert.data.types import MetadataSample

        for dataset in (cfg.train_dataset, cfg.val_dataset):
            provider = instantiate(dataset.resource_providers[0])
            sample = MetadataSample(
                utt_id="19-198-0000",
                path=Path("/original/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"),
                split="train",
            )
            assert provider(sample).value == "/tmp/tts/train-clean-100/19/198/19-198-0000.flac"


def test_training_module_resolves_paired_content_and_original_targets(monkeypatch):
    from quick_convert.components.mixins.resource import ResolvedResource
    from quick_convert.data import AudioBatch
    from quick_convert.training.lightning.modules.ssl_reconstruction import SSLReconstructionTrainingModule

    model = decoder()
    system = SSLReconstructionSystem(model)
    module = SSLReconstructionTrainingModule(system=system, optimization=None)
    monkeypatch.setattr(module, "log_dict", lambda *a, **kw: None)
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        waveforms=torch.randn(2, 2560),
        lengths=torch.tensor([1280, 2560]),
        sample_rates=torch.tensor([16000, 16000]),
        resources={
            "content": ResolvedResource(torch.randn(2, 6, 4), torch.tensor([6, 3])),
            "speaker": ConditioningStream(torch.randn(2, 2, 2), torch.tensor([2, 1]), is_prefix=True),
        },
    )
    output = module._shared_step(batch, "val")
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert module.grad_norm_modules["content_encoder"] is model.content_encoder
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.content_encoder.parameters())
    assert output.lengths.tolist() == [6, 3]
