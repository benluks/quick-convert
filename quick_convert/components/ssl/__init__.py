"""SSL/TLS components for quick-convert.

from .dac import DACContentEncoder
from .emo2vec import EmotionEncoder
from .pros2vec import ProsodyEncoder
from .w2vbert import W2VBertContentEncoder
from .base import ContentEncoder, ContentFeatures

__all__ = [
    "ContentEncoder",
    "ContentFeatures",
    "EmotionEncoder",
    "ProsodyEncoder",
    "W2VBertContentEncoder",
]"""

"""SSL/TLS components for quick-convert.

Lazy imports: each encoder module (and its heavy deps like transformers / funasr /
dac) is only imported when that specific class is first accessed. This lets an
environment that only needs one encoder avoid installing the others' deps.
"""

from importlib import import_module

_LAZY = {
    "DACContentEncoder": ".dac",
    "EmotionEncoder": ".emo2vec",
    "ProsodyEncoder": ".pros2vec",
    "W2VBertContentEncoder": ".w2vbert",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
