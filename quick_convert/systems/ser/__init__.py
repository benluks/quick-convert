"""Speech-emotion recognition systems with lazy optional imports."""

from importlib import import_module

from .base import SERSystem


__all__ = ["SERSystem", "OdysseySER"]


def __getattr__(name: str):
    if name != "OdysseySER":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(".odyssey_ser", __name__), name)


def __dir__():
    return sorted(__all__)
