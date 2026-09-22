import pytest

from quick_convert.components.ssl import (
    CONTENT_ENCODER_ALIASES,
    ContentEncoder,
    S3TokenizerContentEncoder,
    W2VBertContentEncoder,
    resolve_content_encoder,
)


def test_content_encoder_aliases_resolve_without_instantiation():
    assert resolve_content_encoder("w2vbert") is W2VBertContentEncoder
    assert resolve_content_encoder("s3tokenizer") is S3TokenizerContentEncoder


def test_dotted_content_encoder_path_resolves():
    assert (
        resolve_content_encoder("quick_convert.components.ssl.S3TokenizerContentEncoder") is S3TokenizerContentEncoder
    )


def test_content_encoder_class_passes_through():
    assert resolve_content_encoder(S3TokenizerContentEncoder) is S3TokenizerContentEncoder


def test_content_encoder_aliases_are_read_only():
    with pytest.raises(TypeError):
        CONTENT_ENCODER_ALIASES["custom"] = "example.CustomEncoder"


def test_unknown_content_encoder_explains_valid_identifiers():
    with pytest.raises(ValueError, match="dotted class path"):
        resolve_content_encoder("missing")


def test_missing_dotted_content_encoder_has_context():
    with pytest.raises(ImportError, match="Could not resolve content encoder"):
        resolve_content_encoder("quick_convert.components.ssl.MissingEncoder")


def test_resolved_class_must_implement_content_encoder_contract():
    with pytest.raises(TypeError, match="not a ContentEncoder"):
        resolve_content_encoder("pathlib.Path")


def test_identifier_type_is_validated():
    with pytest.raises(TypeError, match="alias, dotted class path"):
        resolve_content_encoder(None)


def test_all_built_in_aliases_resolve_to_content_encoder_classes():
    for alias in CONTENT_ENCODER_ALIASES:
        assert issubclass(resolve_content_encoder(alias), ContentEncoder)
