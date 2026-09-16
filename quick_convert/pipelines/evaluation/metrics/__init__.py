from importlib import import_module

from .base import Metric
from .wer.base import WERMetric


__all__ = ["JiwerWER", "MeanEuclideanDistanceMetric", "Metric", "WERMetric"]

_OPTIONAL_EXPORTS = {
    "JiwerWER": (".wer.jiwer_wer", "JiwerWER"),
    "MeanEuclideanDistanceMetric": (".emo.med", "MeanEuclideanDistanceMetric"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _OPTIONAL_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    attribute = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = attribute
    return attribute
