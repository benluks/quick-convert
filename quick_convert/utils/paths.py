from pathlib import Path
from typing import Any

from ..data.types import AudioSample


class TemplateFormatter:
    @staticmethod
    def format_str(template: str, **context: Any) -> str:
        return template.format(**context)

    @staticmethod
    def format_path(template: str, **context: Any) -> Path:
        return Path(TemplateFormatter.format_str(template, **context)).resolve()

    @staticmethod
    def get_value(obj: Any, key: str) -> Any:
        value = obj

        for part in key.split("."):
            value = value[part] if isinstance(value, dict) else getattr(value, part)

        return value


class SamplePathFormatter(TemplateFormatter):
    @staticmethod
    def format(sample: AudioSample, template: str) -> Path:
        """
        Backwards-compatible path formatter.
        Existing code expects this to return a resolved Path.
        """
        return Path(SamplePathFormatter.format_str(sample, template)).resolve()

    @staticmethod
    def format_str(sample: AudioSample, template: str) -> str:
        """
        String formatter for non-path values like utt_id.
        Supports normal Python format syntax:
          {path.stem}
          {path.parent.name}
          {sample.utt_id}
        """
        return template.format(
            **{
                "sample": sample,
                "path": sample.path,
            }
        )

    @staticmethod
    def _get_sample_value(sample: AudioSample, key: str) -> Any:
        """
        Dot-path resolver for config keys like:
          path.stem
          path.parent.name
          utt_id
          annotations.transcript
        """
        value = sample

        for part in key.split("."):
            if isinstance(value, dict):
                value = value[part]
            else:
                value = getattr(value, part)

        return value
