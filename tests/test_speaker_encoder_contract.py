import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from quick_convert.components.speaker import (
    ESPnetSpeakerEncoder,
    PyannoteWeSpeakerEncoder,
    SpeakerEmbedding,
    SpeakerEncoder,
)
from quick_convert.data import AudioBatch


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def make_batch() -> AudioBatch:
    return AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={},
        waveforms=torch.zeros(2, 8),
        lengths=torch.tensor([8, 5]),
        sample_rates=torch.tensor([16_000, 16_000]),
    )


def test_speaker_embedding_validates_declared_dimension():
    with pytest.raises(ValueError, match="Expected embedding dimension"):
        SpeakerEmbedding(torch.zeros(4), 3, "test", "test")


def test_speaker_encoder_base_loads_files_at_backend_sample_rate(monkeypatch):
    class FakeEncoder(SpeakerEncoder):
        FEATURE_DIM = 3
        sample_rate = 16_000

        def encode(self, wav, sr):
            assert wav.shape == (1, 7)
            assert sr == self.sample_rate
            return SpeakerEmbedding(torch.zeros(3), 3, "fake", "fake")

        def encode_batch(self, samples):
            raise NotImplementedError

    monkeypatch.setattr(
        "quick_convert.components.speaker.speaker_encoders.base.load_audio_input",
        lambda *args, **kwargs: torch.zeros(1, 7),
    )

    assert FakeEncoder(device="cpu").encode_file("input.wav").values.shape == (3,)


def test_espnet_encoder_has_consistent_single_and_batch_contracts(monkeypatch):
    class FakeModel:
        spk_train_args = SimpleNamespace(sample_rate=16_000)

        def __call__(self, waveform):
            assert waveform.shape == (8,)
            return torch.zeros(1, 4)

        def spk_model(self, waveforms, speech_lengths, extract_embd):
            assert extract_embd is True
            assert speech_lengths.tolist() == [8, 5]
            return torch.zeros(waveforms.shape[0], 4)

    class FakeSpeech2Embedding:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "espnet2.bin.spk_inference",
        SimpleNamespace(Speech2Embedding=FakeSpeech2Embedding),
    )

    encoder = ESPnetSpeakerEncoder(device="cpu")
    assert encoder.encode(torch.zeros(1, 8), 16_000).values.shape == (4,)
    assert encoder.encode_batch(make_batch()).values.shape == (2, 4)


def test_pyannote_encoder_has_consistent_single_and_batch_contracts(monkeypatch):
    input_lengths = []

    class FakeModel:
        audio = SimpleNamespace(sample_rate=16_000)
        dimension = 5

        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

    class FakeInference:
        def __init__(self, model, **kwargs):
            pass

        def __call__(self, audio):
            input_lengths.append(audio["waveform"].shape[-1])
            return np.zeros(5, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        SimpleNamespace(Inference=FakeInference, Model=FakeModel),
    )

    encoder = PyannoteWeSpeakerEncoder(device="cpu")
    assert encoder.encode(torch.zeros(1, 8), 16_000).values.shape == (5,)
    assert encoder.encode_batch(make_batch()).values.shape == (2, 5)
    assert input_lengths == [8, 8, 5]


def test_speaker_encoder_configs_compose():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        espnet = compose(config_name="run/precompute_speaker_embedding_espnet_wavlm_joint")
        pyannote = compose(
            config_name="components/speaker_encoder/pyannote_wespeaker_voxceleb_resnet34_LM",
            overrides=["+device=cpu"],
        )

    assert espnet.feature_extractor.encoder is espnet.encoder
    assert espnet.dataset.target_sr == 16_000
    assert pyannote._target_ == "quick_convert.components.speaker.PyannoteWeSpeakerEncoder"
