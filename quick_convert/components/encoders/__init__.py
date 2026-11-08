"""Public encoders components; optional backends load on access."""

from importlib import import_module


_EXPORTS = {
    "ContentEncoder": "quick_convert.components.encoders.content.base",
    "ContentFeatures": "quick_convert.components.encoders.content.base",
    "SpeakerEncoder": "quick_convert.components.encoders.speaker.base",
    "SpeakerEmbedding": "quick_convert.components.encoders.speaker.base",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
