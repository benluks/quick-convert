from __future__ import annotations

from importlib import import_module
from types import MappingProxyType

from quick_convert.components.encoders.content.base import ContentEncoder


CONTENT_ENCODER_ALIASES = MappingProxyType(
    {
        "dac": "quick_convert.components.encoders.content.DACContentEncoder",
        "dac_quantized": "quick_convert.components.encoders.content.DACQuantizedContentEncoder",
        "mimi_quantized": "quick_convert.components.encoders.content.MimiQuantizedContentEncoder",
        "speechtokenizer_quantized": "quick_convert.components.encoders.content.SpeechTokenizerQuantizedContentEncoder",
        "emotion2vec": "quick_convert.components.encoders.content.EmotionEncoder",
        "pase": "quick_convert.components.encoders.content.PASEContentEncoder",
        "paseplus": "quick_convert.components.encoders.content.PASEContentEncoder",
        "spear": "quick_convert.components.encoders.content.SPEARContentEncoder",
        "pros2vec": "quick_convert.components.encoders.content.ProsodyEncoder",
        "s3tokenizer": "quick_convert.components.encoders.content.S3TokenizerContentEncoder",
        "w2vbert": "quick_convert.components.encoders.content.W2VBertContentEncoder",
        "voicefm": "quick_convert.components.encoders.content.VoiceFMContentEncoder",
        "vq_asr": "quick_convert.components.encoders.content.VQASRContentEncoder",
        "wavlm": "quick_convert.components.encoders.content.WavLMContentEncoder",
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
