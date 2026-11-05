"""Audio loading and collation for named resources."""

from dataclasses import replace

from quick_convert.utils.audio import load_audio, load_audio_input

from .base import ResourceRef


def load_audio_resource(ref: ResourceRef, device="cpu"):
    from ..types import AudioSample

    if ref.path is None:
        raise ValueError(f"Resource {ref.name!r} has no path to load.")
    waveform, source_rate = load_audio(ref.path)
    sample_rate = ref.sample_rate if ref.sample_rate is not None else source_rate
    waveform = load_audio_input(
        waveform, sample_rate=source_rate, target_sample_rate=sample_rate, mono=True, device=device
    )
    if waveform.numel() == 0:
        raise ValueError(f"Audio resource {ref.name!r} at {ref.path} is empty.")
    return AudioSample(utt_id=ref.path.stem, path=ref.path, waveform=waveform, sample_rate=sample_rate)


def collate_audio_resources(refs, samples):
    from ..types import AudioBatch, AudioSample

    audio = []
    for ref, sample in zip(refs, samples, strict=True):
        if not isinstance(ref.value, AudioSample) or ref.value.waveform is None:
            raise ValueError(f"Audio resource {ref.name!r} must be loaded as an AudioSample before collation.")
        if ref.value.waveform.ndim not in {1, 2} or (ref.value.waveform.ndim == 2 and ref.value.waveform.shape[0] != 1):
            raise ValueError("Audio resources must contain mono waveforms.")
        if ref.value.waveform.numel() == 0 or ref.value.sample_rate is None or ref.value.sample_rate <= 0:
            raise ValueError("Audio resources require nonempty waveforms and positive sample rates.")
        audio.append(replace(ref.value, utt_id=sample.utt_id, split=sample.split))
    return AudioBatch.from_samples(audio)
