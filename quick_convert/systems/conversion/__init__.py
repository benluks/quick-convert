"""Speech conversion systems and explicit voice requests."""

from importlib import import_module

from .base import SpeechConversionSystem
from .targets import PreparedTarget, ReferenceSpeech, SourceVoice, SpeakerIdentity


__all__ = [
    "SpeechConversionSystem",
    "PreparedTarget",
    "ReferenceSpeech",
    "SourceVoice",
    "SpeakerIdentity",
    "SSLReconstructionConversionSystem",
    "SSLReconstructionResult",
    "KNNVCConversionSystem",
    "ASRBNConversionSystem",
]


def __getattr__(name):
    modules = {
        "SSLReconstructionConversionSystem": "ssl_reconstruction",
        "SSLReconstructionResult": "ssl_reconstruction",
        "KNNVCConversionSystem": "knnvc",
        "ASRBNConversionSystem": "asrbn",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
