from .base import SpeakerEmbedding, SpeakerEncoder
from .campplus import CAMPPlusSpeakerEncoder
from .cosyvoice_campplus import CosyVoiceCAMPPlusSpeakerEncoder
from .espnet import ESPnetSpeakerEncoder
from .pyannote_wespeaker import PyannoteWeSpeakerEncoder


__all__ = [
    "CAMPPlusSpeakerEncoder",
    "CosyVoiceCAMPPlusSpeakerEncoder",
    "ESPnetSpeakerEncoder",
    "PyannoteWeSpeakerEncoder",
    "SpeakerEmbedding",
    "SpeakerEncoder",
]
