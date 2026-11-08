"""Public sequence components; optional backends load on access."""

from importlib import import_module


_EXPORTS = {
    "ConformerEncoder": "quick_convert.components.sequence.conformer_encoder",
    "LayerWeightedConformerEncoder": "quick_convert.components.sequence.conformer_encoder",
    "ParallelLayerWeightedConformerEncoder": "quick_convert.components.sequence.parallel_layer_weighted_conformer",
    "UpsampleConformerEncoder": "quick_convert.components.sequence.upsample_conformer",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
