"""Lazy re-exports.

Each extractor pulls in its own heavy / optional dependencies (e.g. the
tokenizer needs ``sentencepiece``, the speaker extractor needs the speaker
encoder stack). Importing them eagerly here means that any consumer of this
package — including Hydra's ``_target_`` resolution for a *single* extractor —
must have *every* extractor's deps installed. That breaks the isolated, per-task
environments (.venv-dac, .venv-train, ...), where e.g. .venv-dac intentionally
omits ``sentencepiece``.

PEP 562 lazy imports: each name is imported only when first accessed, so
resolving ``...feature_extractors.ContentFeatureExtractor`` no longer drags in
the tokenizer/speaker dependencies.
"""

from importlib import import_module

__all__ = ["ContentFeatureExtractor", "SpeakerEmbeddingExtractor", "TokenizerFeatureExtractor"]

_MODULES = {
    "ContentFeatureExtractor": ".content",
    "SpeakerEmbeddingExtractor": ".speaker_embedding",
    "TokenizerFeatureExtractor": ".tokenizer",
}


def __getattr__(name: str):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
