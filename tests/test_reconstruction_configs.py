from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from quick_convert.components.encoders.content import W2VBertContentEncoder
from quick_convert.utils.resolvers import register_config_resolvers


@pytest.mark.parametrize(
    "run", ["train_sslr_w2vbert_cmdiff", "train_sslr_w2vbert_dit", "train_sslr_w2vbert_cmdiff_rvq"]
)
@pytest.mark.parametrize("frame_hz,ratio", [(50, 1), (25, 2), (12.5, 4)])
def test_reconstruction_configs_keep_content_and_mel_clocks_distinct(monkeypatch, run, frame_hz, ratio):
    register_config_resolvers()
    monkeypatch.setattr(W2VBertContentEncoder, "FRAME_HZ", frame_hz)
    directory = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(config_name=f"run/{run}")
    flow = config.system.decoder.flow
    assert config.train_dataset.target_sr == config.val_dataset.target_sr == 16_000
    assert flow.input_frame_rate == frame_hz
    assert flow.token_mel_ratio == ratio
    assert list(config.trainer.compile.targets) == ["system.decoder.flow"]
    if "encoder" in flow:
        assert flow.encoder.upsample_factor == ratio
        assert flow.encoder._target_ == "quick_convert.components.sequence.upsample_conformer.UpsampleConformerEncoder"


@pytest.mark.parametrize(
    "encoder,content_target,fused_layers",
    [
        ("w2vbert", "W2VBertContentEncoder", 24),
        ("dac", "DACContentEncoder", 1),
        ("speechtokenizer", "SpeechTokenizerQuantizedContentEncoder", None),
    ],
)
@pytest.mark.parametrize("decoder_family", ["dit", "cmdiff"])
def test_campp_recipe_selects_encoder_without_changing_conditioning(
    encoder, content_target, fused_layers, decoder_family
):
    register_config_resolvers()
    directory = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name=f"run/train_sslr_{encoder}_{decoder_family}_campp_prefix", return_hydra_config=True
        )
    choices = config.hydra.runtime.choices
    assert choices.recipe == f"sslr_{decoder_family}_campp_prefix"
    system = config.system
    assert system.online_encoders.content._target_ == f"quick_convert.components.encoders.content.{content_target}"
    assert system.online_encoders.speaker is None
    if fused_layers is None:
        assert system.feature_transform is None
        assert system.online_encoders.content.layer == 0
    else:
        assert system.feature_transform.num_layers == fused_layers

    decoder = system.decoder
    flow = decoder.flow
    assert config.train_dataset.target_sr == config.val_dataset.target_sr == 16_000
    assert decoder.feature_dim == 1024
    assert flow.input_frame_rate == 50
    assert flow.token_mel_ratio == 1
    assert dict(decoder.stream_conditioning.dimensions) == {"frame_speaker": 512}
    assert flow.decoder.estimator.cond_dim == 513
    assert flow.decoder.estimator.spk_dim is None
    if decoder_family == "cmdiff":
        assert flow.encoder.upsample_factor == 1
        assert flow.decoder.estimator.in_channels == 673
