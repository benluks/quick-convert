"""Crop frame features and waveform targets on a shared clock."""

import torch


def aligned_crop(features, frame_lengths, waveforms, sample_lengths, *, hop_size, frames, random=True, positions=None):
    if hop_size <= 0 or frames <= 0:
        raise ValueError("Hop size and crop frames must be positive.")
    crops, targets = [], []
    for i, (x, n, y, m) in enumerate(zip(features, frame_lengths, waveforms, sample_lengths, strict=True)):
        available = min(int(n), int(m) // hop_size)
        if available < frames:
            raise ValueError("Recording is shorter than the aligned crop.")
        if positions is not None:
            start = min(available - frames, int(float(positions[i]) * (available - frames + 1)))
        else:
            start = int(torch.randint(available - frames + 1, ())) if random else 0
        crops.append(x[start : start + frames])
        targets.append(y[start * hop_size : (start + frames) * hop_size])
    return torch.stack(crops), torch.stack(targets)
