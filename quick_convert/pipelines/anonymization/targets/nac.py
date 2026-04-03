from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias


NACRandomTarget: TypeAlias = Literal["random"]

@dataclass(frozen=True)
class NACVoiceTarget:
    speaker_id: str
    voice_dirs: tuple[Path, ...]


NACTarget = NACRandomTarget | NACVoiceTarget


def parse_target(self, raw_target: Any) -> NACTarget:
    if isinstance(raw_target, (NACRandomTarget, NACVoiceTarget)):
        return raw_target

    if raw_target is None or raw_target == "random":
        return NACRandomTarget()

    if not self.voice_dirs:
        raise ValueError(
            "NAC target is a speaker ID, but this anonymizer has no voice_dirs configured."
        )

    return NACVoiceTarget(
        speaker_id=str(raw_target),
        voice_dirs=self.voice_dirs,
    )
