from .base import BaseResourceProvider
from ...utils.paths import SamplePathFormatter


class PathResourceProvider(BaseResourceProvider):
    def __init__(self, name: str, path_template: str, must_exist: bool = True):
        super().__init__(name)
        self.path_template = path_template
        self.must_exist = must_exist

    def __call__(self, sample):
        path = SamplePathFormatter.format(sample, self.path_template)

        if self.must_exist and not path.exists():
            raise FileNotFoundError(f"Missing resource {self.name}: {path}")

        return path
