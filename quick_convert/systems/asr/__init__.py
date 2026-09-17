"""Inference-ready automatic speech recognition systems."""

from .base import ASRSystem
from .vq_asr import VQASRResult, VQASRSystem
from .whisper_asr import WhisperASR


__all__ = ["ASRSystem", "VQASRResult", "VQASRSystem", "WhisperASR"]
