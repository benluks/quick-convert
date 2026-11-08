"""Shared numerical recipes retain their behavior after component reorganization."""

import pytest
import torch


def test_psst1_integrators_preserve_recipe():
    from quick_convert.components.flow_matching.integration import integrate_backward, integrate_forward

    def velocity(x, t):
        return x * 0.125 + t[:, None]

    initial = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    for backward, integrate in [(False, integrate_forward), (True, integrate_backward)]:
        expected = initial.clone()
        for i in range(7):
            t = torch.full((3,), 1.0 - i * (1.0 / 7) if backward else i / 7)
            if backward:
                expected = expected - velocity(expected, t) * (1.0 / 7)
            else:
                expected = expected + velocity(expected, t) * (1.0 / 7)
        assert torch.equal(integrate(velocity, initial, 7), expected)
    assert torch.equal(initial, torch.arange(12, dtype=torch.float32).reshape(3, 4))


@pytest.mark.parametrize("recipe,bins", [("whisper", 80), ("s3tokenizer", 128)])
def test_backend_mel_recipe_is_preserved(recipe, bins):
    from types import SimpleNamespace

    from quick_convert.components.features.mel.s3tokenizer import s3_log_mel
    from quick_convert.components.features.mel.whisper import whisper_log_mel

    waveform = torch.randn(100)
    result = torch.randn(bins, 2)

    def backend_mel(value, *, n_mels):
        assert value is waveform
        assert n_mels == bins
        return result

    extract = whisper_log_mel if recipe == "whisper" else s3_log_mel
    assert extract(waveform, backend=SimpleNamespace(log_mel_spectrogram=backend_mel), n_mels=bins) is result


def test_speaker_cfm_config_constructs_and_runs():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="components/generators/speaker/cfm")
    model = instantiate(cfg.components.generators.speaker)
    vectors = torch.randn(2, 256)
    conditions = torch.randn(2, 512)
    loss = model.compute_loss(vectors, cond=conditions)["loss"]
    assert loss.isfinite()
    generated = model.sample(2, device=torch.device("cpu"), cond=conditions, n_timesteps=2)
    assert generated.shape == vectors.shape
    assert generated.isfinite().all()


def test_all_packaged_run_configs_compose_with_canonical_groups():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for path in sorted((config_dir / "run").glob("*.yaml")):
            compose(config_name="run/" + path.stem)
