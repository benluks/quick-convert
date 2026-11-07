"""Speech-emotion recognition systems with lazy optional imports."""

from importlib import import_module

from .base import SERSystem


__all__ = ["SERSystem", "OdysseySER", "SpeechBrainSER"]

_MODULES = {"OdysseySER": ".odyssey_ser", "SpeechBrainSER": ".speechbrain_ser"}


def __getattr__(name: str):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)


def __dir__():
    return sorted(__all__)
