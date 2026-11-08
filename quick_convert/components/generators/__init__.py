"""Public generators components; optional backends load on access."""

from importlib import import_module


_EXPORTS = {
    "PSST1FlowSpeakerGenerator": "quick_convert.components.generators.speaker.psst1_flow",
    "CFMSpeakerGenerator": "quick_convert.components.generators.speaker.cfm",
    "KnnVCHifiGanDecoder": "quick_convert.components.generators.waveform.knnvc_hifigan",
    "PSST1HifiGanDecoder": "quick_convert.components.generators.waveform.psst1_hifigan",
    "CosyVoiceSpectrogramGenerator": "quick_convert.components.generators.acoustic.cosyvoice",
    "CosyVoiceHiFTDecoder": "quick_convert.components.generators.waveform.hift_generator",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
