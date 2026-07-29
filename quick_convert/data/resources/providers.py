from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from quick_convert.data.types import AudioSample

from pathlib import Path

from quick_convert.components.feature_extractors.base import BaseFeatureExtractor

from ...utils.paths import SamplePathFormatter
from .base import BaseResourceProvider, ResourceRef


class TemplateResourceProvider(BaseResourceProvider):
    def __init__(self, name: str, template: str, kind: str = "text"):
        super().__init__(name)
        self.template = template
        self.kind = kind

    def resolve(self, sample):
        return SamplePathFormatter.format_str(sample, self.template)

    def __call__(self, sample):
        return ResourceRef(name=self.name, kind=self.kind, value=self.resolve(sample))


class PathResourceProvider(TemplateResourceProvider):
    def __init__(
        self,
        name,
        path_template,
        kind,
        # only set `max_length` for tensor resources
        max_length=None,
        must_exist=True,
    ):
        super().__init__(name=name, template=path_template, kind=kind)
        self.max_length = max_length
        self.must_exist = must_exist

    def __call__(self, sample):
        path = Path(self.resolve(sample))

        if self.must_exist and not path.exists():
            raise FileNotFoundError(f"Missing resource {self.name}: {path}")

        return ResourceRef(name=self.name, kind=self.kind, path=path, value=None, max_length=self.max_length)


class OnlineFeatureProvider(BaseResourceProvider):
    """
    Materialize a feature resource when a dataset sample is loaded.

    Unlike path-based resource providers, this provider computes the resource
    immediately using a feature extractor and stores the resulting value
    directly in the returned `ResourceRef`.

    The sample must already contain all inputs required by the extractor. For
    an audio feature extractor, this normally means that the dataset must load
    the waveform before its resource providers are evaluated.

    Notes
    -----
    This provider is executed inside `Dataset.__getitem__`. Consequently, when
    a dataloader uses worker processes, each worker receives its own copy of the
    feature extractor. It is therefore best suited to lightweight, CPU-based
    extraction.

    Large neural feature extractors, especially those intended to run on a GPU,
    should generally be applied after batching or used during preprocessing.
    Using such an extractor here may duplicate models across workers and repeat
    computation whenever the same sample is accessed.

    Parameters
    ----------
    extractor:
        Feature extractor used to compute the resource from a sample.
    name:
        Name under which the resource is exposed. When omitted,
        `extractor.feature_name` is used.
    kind:
        Resource kind assigned to the resulting `ResourceRef`.
    """

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        name: str | None = None,
        kind: str = "torch_tensor",
    ) -> None:
        resolved_name = extractor.feature_name if name is None else name

        super().__init__(resolved_name)

        self.extractor = extractor
        self.kind = kind

    def __call__(self, sample: AudioSample) -> ResourceRef:
        value = self.extractor.extract_sample(sample)

        return ResourceRef(
            name=self.name,
            kind=self.kind,
            value=value,
        )
