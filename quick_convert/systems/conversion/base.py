"""Structural contract for systems used by dataset conversion workflows."""

from typing import Protocol

from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.anonymization.targets import PreparedTarget, ReferenceSpeech


class SpeechConversionSystem(Protocol):
    """Prepare backend-specific voice conditioning and convert source batches.

    Target selection is a workflow concern. Systems validate the representation
    and model provenance of prepared conditioning before applying it.
    """

    def prepare_target(self, target: ReferenceSpeech | PreparedTarget) -> PreparedTarget: ...

    def convert_batch(self, batch: AudioBatch, *, target: PreparedTarget) -> GeneratedAudio: ...
