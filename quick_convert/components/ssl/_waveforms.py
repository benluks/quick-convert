from __future__ import annotations

import torch


def validate_waveforms(waveforms, lengths, sample_rate, expected_sample_rate):
    """Validate padded mono audio without changing caller-owned tensors."""
    if sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(f"Expected {expected_sample_rate} Hz audio, got {sample_rate} Hz.")
    if waveforms.ndim != 2 or waveforms.shape[0] == 0:
        raise ValueError("Expected a nonempty waveform batch with shape (batch, samples).")
    if not waveforms.is_floating_point():
        raise TypeError("Waveforms must use a floating-point dtype.")
    if lengths is None:
        lengths = torch.full((waveforms.shape[0],), waveforms.shape[1], dtype=torch.long, device=waveforms.device)
    if lengths.shape != (waveforms.shape[0],):
        raise ValueError("Expected one waveform length per batch item.")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("Waveform lengths must be integer sample counts.")
    if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[1]):
        raise ValueError("Waveform lengths must be positive and cannot exceed the padded waveform size.")
    return lengths
