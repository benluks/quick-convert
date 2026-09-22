from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

from .base import ContentEncoder


CONTENT_ENCODER_ALIASES = MappingProxyType(
    {
        "dac": "quick_convert.components.ssl.DACContentEncoder",
        "emotion2vec": "quick_convert.components.ssl.EmotionEncoder",
        "pros2vec": "quick_convert.components.ssl.ProsodyEncoder",
        "s3tokenizer": "quick_convert.components.ssl.S3TokenizerContentEncoder",
        "w2vbert": "quick_convert.components.ssl.W2VBertContentEncoder",
        "wavlm": "quick_convert.components.ssl.WavLMContentEncoder",
    }
)


def resolve_content_encoder(identifier: str | type[ContentEncoder]) -> type[ContentEncoder]:
    """Resolve a content encoder alias or dotted class path.

    Passing an encoder class is also supported, which lets callers normalize
    configuration values without special-casing already-resolved classes.
    Resolution imports the class but does not construct it or load model
    weights.
    """
    if isinstance(identifier, type):
        encoder_class = identifier
    elif isinstance(identifier, str):
        dotted_path = CONTENT_ENCODER_ALIASES.get(identifier, identifier)
        module_name, separator, attribute_name = dotted_path.rpartition(".")
        if not separator or not module_name or not attribute_name:
            aliases = ", ".join(sorted(CONTENT_ENCODER_ALIASES))
            raise ValueError(f"Unknown content encoder {identifier!r}. Use a dotted class path or one of: {aliases}.")

        try:
            module = import_module(module_name)
            encoder_class = getattr(module, attribute_name)
        except (ImportError, AttributeError) as error:
            raise ImportError(f"Could not resolve content encoder {identifier!r} from {dotted_path!r}.") from error
    else:
        raise TypeError("A content encoder identifier must be an alias, dotted class path, or ContentEncoder class.")

    if not isinstance(encoder_class, type) or not issubclass(encoder_class, ContentEncoder):
        raise TypeError(f"Resolved object {encoder_class!r} is not a ContentEncoder class.")
    return encoder_class
