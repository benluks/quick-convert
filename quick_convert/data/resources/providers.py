from pathlib import Path

from ...utils.paths import SamplePathFormatter
from .base import BaseResourceProvider, ResourceKind, ResourceRef


class TemplateResourceProvider(BaseResourceProvider):
    def __init__(self, name: str, template: str, kind: ResourceKind = "text"):
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
        kind: ResourceKind,
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
