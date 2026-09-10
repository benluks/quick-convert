from .cosyvoice import CosyVoiceDecoderOutput, CosyVoiceGenerationOutput, CosyVoiceSpectrogramGenerator


__all__ = [
    "CosyVoiceDecoderOutput",
    "CosyVoiceGenerationOutput",
    "CosyVoiceHiFTDecoder",
    "CosyVoiceSpectrogramGenerator",
]


def __getattr__(name: str):
    if name == "CosyVoiceHiFTDecoder":
        from .hift_generator import CosyVoiceHiFTDecoder

        return CosyVoiceHiFTDecoder
    raise AttributeError(name)
