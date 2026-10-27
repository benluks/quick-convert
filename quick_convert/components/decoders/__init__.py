from .cosyvoice import CosyVoiceDecoderOutput, CosyVoiceGenerationOutput, CosyVoiceSpectrogramGenerator


__all__ = [
    "CosyVoiceDecoderOutput",
    "CosyVoiceGenerationOutput",
    "CosyVoiceHiFTDecoder",
    "CosyVoiceSpectrogramGenerator",
    "KnnVCHifiGanDecoder",
    "PSST1Vocoder",
]


def __getattr__(name: str):
    if name == "CosyVoiceHiFTDecoder":
        from .hift_generator import CosyVoiceHiFTDecoder

        return CosyVoiceHiFTDecoder
    if name == "KnnVCHifiGanDecoder":
        from .knnvc_hifigan import KnnVCHifiGanDecoder

        return KnnVCHifiGanDecoder
    if name == "PSST1Vocoder":
        from .psst1_vocoder import PSST1Vocoder

        return PSST1Vocoder
    raise AttributeError(name)
