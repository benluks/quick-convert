from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torchaudio

from ...data.types import AudioBatch, MetadataSample
from .base import ASRSystem


_WHISPER_SR = 16000


class WhisperASR(ASRSystem):
    def __init__(
        self,
        device,
        model_name: str = "base.en",
        language: str = "en",
        sr: int = _WHISPER_SR,
        pred_key="transcript",
        name="whisper",
    ):
        super().__init__(name, sr, device, pred_key=pred_key)
        self.model_name = model_name
        self.language = language
        self._model = None

        try:
            import whisper
        except ImportError as error:
            raise ImportError("Whisper ASR requires the `whisper` extra.") from error

        self._whisper = whisper
        self.decoding_options = whisper.DecodingOptions(language=language)

        if self.device == "mps":
            self.device = "cpu"

    def _get_model(self):
        if self._model is None:
            self._model = self._whisper.load_model(self.model_name).to(self.device)
        return self._model

    def transcribe(self, sample: MetadataSample) -> str:
        model = self._get_model()
        waveform, sr = torchaudio.load(str(sample.path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != _WHISPER_SR:
            waveform = torchaudio.functional.resample(waveform, sr, _WHISPER_SR)
        audio = waveform.squeeze().numpy()
        result = model.transcribe(audio, language=self.language)
        return result["text"]

    def transcribe_batch(self, batch: AudioBatch) -> list[str]:
        if batch.waveforms is None or batch.sample_rates is None:
            raise ValueError("WhisperASR requires a batch with loaded audio.")
        assert (batch.sample_rates == self.sr).all()
        model = self._get_model()
        wav = self._whisper.pad_or_trim(batch.waveforms.to(self.device))
        mel = self._whisper.log_mel_spectrogram(wav)
        decoding_results: Iterable[Any] = model.decode(mel, self.decoding_options)

        return [res.text for res in decoding_results]
