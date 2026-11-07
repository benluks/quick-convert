from importlib import import_module

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


_OPTIONAL_EXPORTS = {
    "CosyVoiceHiFTDecoder": ".hift_generator",
    "KnnVCHifiGanDecoder": ".knnvc_hifigan",
    "PSST1HifiGanDecoder": ".psst1_hifigan",
}


def __getattr__(name: str):
    try:
        module = _OPTIONAL_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    attribute = getattr(import_module(module, __name__), name)
    globals()[name] = attribute
    return attribute
