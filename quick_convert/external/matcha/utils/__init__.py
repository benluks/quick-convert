"""Matcha utility exports, loaded only when requested.

The supported Quick Convert decoder imports ``matcha.utils.audio`` directly and
must not acquire Matcha's training and CLI dependencies as a side effect.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "instantiate_callbacks": ("matcha.utils.instantiators", "instantiate_callbacks"),
    "instantiate_loggers": ("matcha.utils.instantiators", "instantiate_loggers"),
    "log_hyperparameters": ("matcha.utils.logging_utils", "log_hyperparameters"),
    "get_pylogger": ("matcha.utils.pylogger", "get_pylogger"),
    "enforce_tags": ("matcha.utils.rich_utils", "enforce_tags"),
    "print_config_tree": ("matcha.utils.rich_utils", "print_config_tree"),
    "extras": ("matcha.utils.utils", "extras"),
    "get_metric_value": ("matcha.utils.utils", "get_metric_value"),
    "task_wrapper": ("matcha.utils.utils", "task_wrapper"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
