from .cosyvoice import CosyVoiceDecoderOutput, CosyVoiceGenerationOutput, CosyVoiceSpectrogramGenerator


__all__ = [
    "CosyVoiceDecoderOutput",
    "CosyVoiceGenerationOutput",
    "CosyVoiceHiFTDecoder",
    "CosyVoiceSpectrogramGenerator",
    "KnnVCHifiGanDecoder",
]


def __getattr__(name: str):
    if name == "CosyVoiceHiFTDecoder":
        from .hift_generator import CosyVoiceHiFTDecoder

        return CosyVoiceHiFTDecoder
    if name == "KnnVCHifiGanDecoder":
        from .knnvc_hifigan import KnnVCHifiGanDecoder

        return KnnVCHifiGanDecoder
    raise AttributeError(name)
