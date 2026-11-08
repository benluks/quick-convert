"""Public heads components; optional backends load on access."""

from importlib import import_module


_EXPORTS = {
    "SupervisedHead": "quick_convert.components.heads.base",
    "HeadOutput": "quick_convert.components.heads.base",
    "HeadTarget": "quick_convert.components.heads.base",
    "LinearHead": "quick_convert.components.heads.linear_head",
    "LinguisticCTCHead": "quick_convert.components.heads.linguistic_head",
    "SpeakerASPHead": "quick_convert.components.heads.speaker_head",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
