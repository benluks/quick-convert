"""Read CosyVoice companion configuration as data, without constructing YAML tags."""

from pathlib import Path

from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig


def read_cosyvoice_config(path: str | Path) -> tuple[dict, dict, MelSpectrogramConfig]:
    """Read HiFT architecture and preprocessing fields; reject unresolved expressions."""
    import yaml

    root = yaml.compose(Path(path).read_text(), Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        raise ValueError("CosyVoice configuration must be a mapping.")
    nodes = {key.value: value for key, value in root.value}

    def read(node):
        if node.tag == "!ref":
            name = node.value.strip()
            if not (name.startswith("<") and name.endswith(">") and name.count("<") == 1):
                raise ValueError(f"Unsupported configuration reference: {name!r}")
            referenced = nodes.get(name[1:-1])
            if referenced is None or referenced.tag == "!ref":
                raise ValueError(f"Unresolved configuration reference: {name!r}")
            return read(referenced)
        if not (node.tag.startswith("tag:yaml.org,2002:") or node.tag.startswith(("!new:", "!name:"))):
            raise ValueError(f"Unsupported configuration tag: {node.tag!r}")
        if isinstance(node, yaml.MappingNode):
            return {key.value: read(value) for key, value in node.value}
        if isinstance(node, yaml.SequenceNode):
            return [read(value) for value in node.value]
        if isinstance(node, yaml.ScalarNode) and node.tag.startswith("tag:yaml.org,2002:"):
            return yaml.safe_load(yaml.serialize(node))
        raise ValueError(f"Unsupported configuration tag: {node.tag!r}")

    def require_mapping(node, tag):
        if not isinstance(node, yaml.MappingNode) or node.tag not in {"tag:yaml.org,2002:map", tag}:
            raise ValueError(f"Unsupported companion component; expected {tag!r}.")

    try:
        require_mapping(nodes["hift"], "!new:cosyvoice.hifigan.generator.CausalHiFTGenerator")
        require_mapping(nodes["feat_extractor"], "!name:matcha.utils.audio.mel_spectrogram")
        hift_nodes = {key.value: value for key, value in nodes["hift"].value}
        require_mapping(hift_nodes["f0_predictor"], "!new:cosyvoice.hifigan.f0_predictor.CausalConvRNNF0Predictor")
        hift = read(nodes["hift"])
        f0 = hift.pop("f0_predictor")
        mel = read(nodes["feat_extractor"])
        mel["sample_rate"] = mel.pop("sampling_rate")
        return hift, f0, MelSpectrogramConfig(**mel)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Companion configuration must declare HiFT and complete mel preprocessing settings."
        ) from error
