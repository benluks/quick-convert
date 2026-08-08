# quick_convert/data/resources/loading.py
# or factory.py, if you want to avoid confusing this with resource file loading

from hydra.utils import instantiate

from quick_convert.utils.config import compose_component

from .providers import BaseResourceProvider


def load_resource_provider(
    name: str,
    **overrides,
) -> BaseResourceProvider:
    """Instantiate a packaged resource-provider recipe.

    Args:
        name:
            Name of a config in the ``resource_provider`` config group.
        **overrides:
            Values to override in the packaged provider recipe.

    Examples:
        Load a provider as defined::

            provider = load_resource_provider(
                "librispeech_speaker",
            )

        Override part of the recipe::

            provider = load_resource_provider(
                "wavlm_features",
                path_template="/tmp/wavlm/{sample.utt_id}.pt",
            )
    """

    cfg = compose_component(
        group="resource_providers",
        name=name,
        overrides=overrides,
    )

    provider = instantiate(cfg)

    if not isinstance(provider, BaseResourceProvider):
        raise TypeError(
            f"Resource provider recipe {name!r} produced {type(provider).__name__}, expected BaseResourceProvider."
        )

    return provider
