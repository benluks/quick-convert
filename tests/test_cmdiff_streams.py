"""CMDiff shares prefix conditions, continuation loss and no-x-vector inference."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from quick_convert.components.decoders import ConditioningStream, CosyVoiceSpectrogramGenerator, StreamConditioning
from quick_convert.components.encoders.upsample_conformer import UpsampleConformerEncoder
from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig


CausalConditionalDecoder = pytest.importorskip("quick_convert.external.cosyvoice.flow.decoder").CausalConditionalDecoder
CausalMaskedDiffWithXvec = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow").CausalMaskedDiffWithXvec
CausalConditionalCFM = pytest.importorskip("quick_convert.external.cosyvoice.flow.flow_matching").CausalConditionalCFM


@pytest.mark.parametrize("channels", [(32,), (32, 32)])
def test_cmdiff_real_flow_trains_and_generates_with_nonmel_prefixes(monkeypatch, channels):
    torch.manual_seed(5)
    estimator = CausalConditionalDecoder(
        in_channels=23,
        out_channels=8,
        channels=channels,
        attention_head_dim=16,
        num_heads=2,
        n_blocks=1,
        num_mid_blocks=1,
        dropout=0,
        act_fn="gelu",
        cond_dim=7,
        spk_dim=None,
    )
    cfm = CausalConditionalCFM(
        in_channels=16,
        estimator=estimator,
        cfm_params=DictConfig(
            {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "cosine",
                "training_cfg_rate": 0.2,
                "inference_cfg_rate": 0.7,
            }
        ),
    )
    encoder = UpsampleConformerEncoder(
        input_size=8,
        output_size=16,
        num_blocks=1,
        post_upsample_blocks=1,
        attention_heads=2,
        linear_units=32,
        upsample_factor=2,
        dropout_rate=0,
        positional_dropout_rate=0,
        attention_dropout_rate=0,
        use_cnn_module=False,
        macaron_style=False,
    )
    flow = CausalMaskedDiffWithXvec(
        input_size=8,
        output_size=8,
        vocab_size=None,
        input_frame_rate=25,
        token_mel_ratio=2,
        encoder=encoder,
        decoder=cfm,
    )
    model = CosyVoiceSpectrogramGenerator(
        flow,
        feature_dim=8,
        mel_dim=8,
        device="cpu",
        stream_inference_steps=2,
        mel_config=MelSpectrogramConfig(16000, 1024, 320, 1024, 8, 0, None),
        stream_conditioning=StreamConditioning(
            {"speaker": 3, "prosody": 2},
            prefix_fraction=(0.5, 0.5),
            context_mask_fraction=(0, 0),
            stream_drop_probability=0,
        ),
    )
    assert not model.requires_speaker_embedding
    assert flow.spk_embed_affine_layer is None
    features = torch.randn(2, 5, 8, requires_grad=True)
    lengths = torch.tensor([5, 3])
    mel_lengths = lengths * 2
    speaker = torch.randn(2, 10, 3, requires_grad=True)
    prosody = torch.randn(2, 10, 2, requires_grad=True)
    original = speaker.detach().clone()
    monkeypatch.setattr(model, "_compute_mels", lambda *a: (torch.randn(2, 8, 10), mel_lengths))
    output = model.compute_loss(
        features,
        lengths,
        torch.zeros(2, 3200),
        mel_lengths * 320,
        16000,
        streams={
            "speaker": ConditioningStream(speaker, mel_lengths),
            "prosody": ConditioningStream(prosody, mel_lengths),
        },
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert torch.isfinite(features.grad).all()
    assert features.grad[0].abs().sum() > 0
    assert speaker.grad[0, :5].abs().sum() > 0
    assert speaker.grad[0, 5:].count_nonzero() == 0
    assert prosody.grad[1, 3:].count_nonzero() == 0
    torch.testing.assert_close(speaker, original)
    generated = model(
        features.detach(),
        lengths,
        streams={
            "speaker": ConditioningStream(speaker.detach(), torch.tensor([3, 1])),
            "prosody": ConditioningStream(prosody.detach(), torch.tensor([1, 2])),
        },
    )
    assert generated.mel.shape == (2, 8, 10)
    assert generated.mel_lengths.tolist() == [10, 6]
    assert generated.mel[1, :, 6:].count_nonzero() == 0
    assert torch.isfinite(generated.mel).all()


def test_cmdiff_stream_run_composes_and_estimator_accepts_odd_condition_width():
    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name="run/train_sslr_w2vbert_cmdiff_streams_no_xvector")
        assert cfg.system.online_encoders.speaker is None
        conditioning = instantiate(cfg.system.decoder.stream_conditioning)
        estimator_cfg = cfg.system.decoder.flow.decoder.estimator
        assert estimator_cfg.in_channels == 160 + conditioning.output_dim == 241
        assert estimator_cfg.spk_dim is None
        estimator = instantiate(
            estimator_cfg, channels=[32], n_blocks=1, num_mid_blocks=1, num_heads=2, attention_head_dim=16
        )
        assert estimator.time_embeddings.dim == 242
    with pytest.raises(ValueError, match="in_channels"):
        CausalConditionalDecoder(in_channels=22, out_channels=8, cond_dim=7, spk_dim=None)
