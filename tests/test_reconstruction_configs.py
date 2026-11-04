from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from quick_convert.components.ssl import W2VBertContentEncoder
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
        assert flow.encoder._target_ == "quick_convert.components.encoders.upsample_conformer.UpsampleConformerEncoder"
