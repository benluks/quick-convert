from pathlib import Path
from typing import Callable

from .base import BaseResourceProvider, ResourceRef
from ...utils.paths import SamplePathFormatter


class TemplateResourceProvider(BaseResourceProvider):
    def __init__(
        self,
        name: str,
        template: str,
        kind: str = "text",
    ):
        super().__init__(name)
        self.template = template
        self.kind = kind

    def resolve(self, sample):
        return SamplePathFormatter.format(sample, self.path_template)

    def __call__(self, sample):
        return ResourceRef(name=self.name, kind=self.kind, value=self.resolve(sample))


class PathResourceProvider(TemplateResourceProvider):
    def __init__(
        self,
        name,
        path_template,
        kind,
        must_exist=True,
    ):
        super().__init__(
            name=name,
            template=path_template,
            kind=kind,
        )
        self.must_exist = must_exist

    def __call__(self, sample):
        path = self.resolve(sample)

        if self.must_exist and not path.exists():
            raise FileNotFoundError(f"Missing resource {self.name}: {path}")

        return ResourceRef(name=self.name, kind=self.kind, path=path, value=None)
