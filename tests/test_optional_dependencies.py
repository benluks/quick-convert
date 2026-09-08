import sys
from types import SimpleNamespace

import torch

from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.tokenizer.bpe import SentencePieceBPETrainer
from quick_convert.systems.asr.whisper_asr import WhisperASR


def test_jiwer_module_remains_available_after_construction(monkeypatch):
    fake_jiwer = SimpleNamespace(
        Compose=lambda transforms: transforms,
        ToLowerCase=lambda: object(),
        RemovePunctuation=lambda: object(),
        Strip=lambda: object(),
        ReduceToListOfListOfWords=lambda: object(),
        wer=lambda references, hypotheses, **kwargs: 0.25,
    )
    monkeypatch.setitem(sys.modules, "jiwer", fake_jiwer)

    from quick_convert.pipelines.evaluation.metrics import JiwerWER

    metric = JiwerWER()

    assert metric.compute("reference", "hypothesis") == {"wer": 0.25}


def test_sentencepiece_module_remains_available_after_construction(monkeypatch, tmp_path):
    trained_with = {}

    class FakeTrainer:
        @staticmethod
        def train(**kwargs):
            trained_with.update(kwargs)

    class FakeProcessor:
        def load(self, path):
            self.path = path

        def get_piece_size(self):
            return 17

    fake_sentencepiece = SimpleNamespace(
        SentencePieceTrainer=FakeTrainer,
        SentencePieceProcessor=FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", fake_sentencepiece)

    trainer = SentencePieceBPETrainer(vocab_size=17)
    trainer._train(model_prefix="tokenizer")
    trainer._model = trainer._load(tmp_path / "tokenizer.model")

    assert trained_with["model_prefix"] == "tokenizer"
    assert trainer.vocab_size_actual == 17


def test_whisper_batch_path_uses_retained_optional_module(monkeypatch):
    class FakeModel:
        def to(self, device):
            return self

        def decode(self, mel, options):
            return [SimpleNamespace(text="hello")]

    fake_whisper = SimpleNamespace(
        DecodingOptions=lambda **kwargs: kwargs,
        load_model=lambda name: FakeModel(),
        pad_or_trim=lambda waveform: waveform,
        log_mel_spectrogram=lambda waveform: waveform,
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    system = WhisperASR(device="cpu")
    batch = AudioBatch(
        utt_ids=["sample"],
        paths=[],
        splits=[None],
        resources={},
        waveforms=torch.zeros(1, 16),
        lengths=torch.tensor([16]),
        sample_rates=torch.tensor([16_000]),
    )

    assert system.get_labels(batch) == ["hello"]
