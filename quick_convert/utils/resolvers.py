"""OmegaConf resolvers used by installed Quick Convert configurations."""

import math
from pathlib import Path
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


def frame_ratio(output_hz: float, input_hz: float) -> int:
    """Require an integer temporal upsampling ratio without silently rounding."""
    if not all(math.isfinite(rate) and rate > 0 for rate in (output_hz, input_hz)):
        raise ValueError("Frame rates must be finite and positive.")
    ratio = output_hz / input_hz
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-6):
        raise ValueError(f"Unsupported frame rates: {input_hz} Hz to {output_hz} Hz requires noninteger upsampling.")
    return round(ratio)


def register_config_resolvers() -> None:
    """Register the resolvers required by packaged configurations."""
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": bool,
        "len": len,
        "class_attr": class_attribute,
        "frame_ratio": frame_ratio,
        "conversion_dir": conversion_directory,
        "path_stem": lambda path: Path(path).stem,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def conversion_directory(root, metadata, target, suffix="") -> str:
    """Resolve conversion layouts without importing inference backends."""
    from quick_convert.pipelines.conversion_records import conversion_output_dir

    return conversion_output_dir(root, metadata, target, suffix)
