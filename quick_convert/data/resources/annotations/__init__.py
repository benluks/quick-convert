# annotations are resources that are metadata of the dataset, and not computed features
from .transcript import CSVTranscriptProvider
# from .speaker_id import CSVSpeakerIDProvider

__all__ = ["CSVTranscriptProvider"]
