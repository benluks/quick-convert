"""OmegaConf resolvers used by installed Quick Convert configurations."""

from typing import Any

from hydra.utils import get_class
from omegaconf import OmegaConf


def class_attribute(target: str, attribute: str) -> Any:
    """Read a public constant from a configured class without instantiating it."""
    if not attribute.isupper() or attribute.startswith("_"):
        raise ValueError("class_attr only exposes public uppercase capability constants.")

    cls = get_class(target)
    try:
        value = getattr(cls, attribute)
    except AttributeError as error:
        raise ValueError(f"{target} does not define capability {attribute}.") from error
    if callable(value):
        raise TypeError(f"Capability {target}.{attribute} must be a value, not a callable.")
    return value


def register_config_resolvers() -> None:
    """Register the resolvers required by packaged configurations."""
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": bool,
        "len": len,
        "class_attr": class_attribute,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
