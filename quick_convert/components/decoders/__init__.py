from .cosyvoice import CosyVoiceDecoderOutput, CosyVoiceGenerationOutput, CosyVoiceSpectrogramGenerator
from .streams import ConditioningStream, StreamConditioning


__all__ = [
    "ConditioningStream",
    "StreamConditioning",
    "CosyVoiceDecoderOutput",
    "CosyVoiceGenerationOutput",
    "CosyVoiceHiFTDecoder",
    "CosyVoiceSpectrogramGenerator",
    "KnnVCHifiGanDecoder",
    "PSST1HifiGanDecoder",
]


def __getattr__(name: str):
    if name == "CosyVoiceHiFTDecoder":
        from .hift_generator import CosyVoiceHiFTDecoder

        return CosyVoiceHiFTDecoder
    if name == "KnnVCHifiGanDecoder":
        from .knnvc_hifigan import KnnVCHifiGanDecoder

        return KnnVCHifiGanDecoder
    if name == "PSST1HifiGanDecoder":
        from .psst1_hifigan import PSST1HifiGanDecoder

        return PSST1HifiGanDecoder
    raise AttributeError(name)
