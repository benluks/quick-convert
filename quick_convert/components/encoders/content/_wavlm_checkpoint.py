"""Load Microsoft's original WavLM-Large weights into Transformers."""

import re
from pathlib import Path

import torch


# The Google Drive download linked in Microsoft's WavLM README.
_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU&export=download&confirm=t"
)
_CACHE_NAME = "microsoft-wavlm-large-6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f.pt"

# Adapt the parameter names from the upstream Transformers conversion script.
_KEY_REPLACEMENTS = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.": "attention.",
    "attention.grep_linear": "attention.gru_rel_pos_linear",
    "attention.relative_attention_bias": "attention.rel_attn_embed",
    "attention.grep_a": "attention.gru_rel_pos_const",
    "self_attn_layer_norm": "layer_norm",
    ".fc1.": ".feed_forward.intermediate_dense.",
    ".fc2.": ".feed_forward.output_dense.",
    "mask_emb": "masked_spec_embed",
    "weight_g": "parametrizations.weight.original0",
    "weight_v": "parametrizations.weight.original1",
}


def _transformers_key(key: str) -> str:
    if key.startswith("layer_norm."):
        return f"feature_projection.{key}"
    key = re.sub(r"(conv_layers\.\d+)\.0\.", r"\1.conv.", key)
    key = re.sub(r"(conv_layers\.\d+)\.2\.1\.", r"\1.layer_norm.", key)
    for original, converted in _KEY_REPLACEMENTS.items():
        key = key.replace(original, converted)
    return key


def load_microsoft_wavlm_large(model: torch.nn.Module, *, local_files_only: bool) -> None:
    """Load encoder parameters and reject missing or incompatible weights."""
    if local_files_only:
        path = Path(torch.hub.get_dir()) / "checkpoints" / _CACHE_NAME
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.hub.load_state_dict_from_url(
            _CHECKPOINT_URL,
            file_name=_CACHE_NAME,
            check_hash=True,
            map_location="cpu",
            weights_only=True,
        )

    expected = model.state_dict().keys()
    weights = {_transformers_key(key): value for key, value in checkpoint["model"].items()}
    # The original checkpoint also contains pretraining parameters.
    model.load_state_dict({key: value for key, value in weights.items() if key in expected}, strict=True)
