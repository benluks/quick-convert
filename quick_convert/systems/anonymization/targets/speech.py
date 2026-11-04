"""Target requests shared by speech conversion systems.

Requests describe evidence for a voice. Prepared targets contain backend-specific
conditioning; their compatibility is checked by the system that consumes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar


if TYPE_CHECKING:
    from quick_convert.components.reference_prompt import PromptSegment, SpeechPromptBuilder
    from quick_convert.components.speaker.aggregation import SpeakerEmbeddingAggregation


ConditioningT = TypeVar("ConditioningT")


@dataclass(frozen=True)
class ReferenceSpeech:
    """Reference recordings with optional aggregation and acoustic prompting.

    Backends choose their default preparation when aggregation is None and
    reject unsupported options. prompt_seconds requests an acoustic prefix;
    prompt_builder can assemble active speech across recordings. SSL stream
    models can explicitly allow unavailable reference extractors to be omitted.
    """

    allow_missing_streams: bool
    paths: tuple[Path, ...]
    aggregation: SpeakerEmbeddingAggregation | None
    prompt_seconds: float | None
    prompt_builder: SpeechPromptBuilder | None

    def __init__(
        self,
        paths: list[str | PathLike] | tuple[str | PathLike, ...],
        *,
        allow_missing_streams: bool = False,
        aggregation: SpeakerEmbeddingAggregation | None = None,
        prompt_seconds: float | None = None,
        prompt_builder: SpeechPromptBuilder | None = None,
    ):
        if isinstance(paths, (str, PathLike)) or not paths:
            raise ValueError("ReferenceSpeech requires a nonempty sequence of recording paths.")
        if type(allow_missing_streams) is not bool:
            raise TypeError("allow_missing_streams must be a boolean.")
        object.__setattr__(self, "allow_missing_streams", allow_missing_streams)
        if aggregation is not None:
            from quick_convert.components.speaker.aggregation import SpeakerEmbeddingAggregation

            if not isinstance(aggregation, SpeakerEmbeddingAggregation):
                raise TypeError("aggregation must be a SpeakerEmbeddingAggregation policy.")
        if prompt_seconds is not None and (not math.isfinite(prompt_seconds) or prompt_seconds <= 0):
            raise ValueError("prompt_seconds must be finite and positive.")
        if prompt_builder is not None:
            from quick_convert.components.reference_prompt import SpeechPromptBuilder

            if not isinstance(prompt_builder, SpeechPromptBuilder):
                raise TypeError("prompt_builder must be a SpeechPromptBuilder.")
            if prompt_seconds is None:
                raise ValueError("prompt_builder requires prompt_seconds.")
        object.__setattr__(self, "prompt_builder", prompt_builder)
        object.__setattr__(self, "paths", tuple(Path(path) for path in paths))
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(self, "prompt_seconds", prompt_seconds)


@dataclass(frozen=True)
class PreparedTarget(Generic[ConditioningT]):
    """Reusable conditioning in an explicitly identified model space.

    Matching dimensions alone do not imply compatibility. ``space`` identifies
    the producing model, and ``representation`` identifies the payload contract.
    Systems must validate both before using ``conditioning``.
    """

    conditioning: ConditioningT
    space: str
    representation: str
    references: tuple[Path, ...] = ()
    aggregation: str | None = None
    durations: tuple[float, ...] = ()
    prompt_segments: tuple[PromptSegment, ...] = ()
    prompt_sample_rate: int | None = None
    prompt_crossfade_seconds: float | None = None
