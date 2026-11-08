"""OpenAI Whisper mel recipe, delegated to the selected backend version.

The backend owns filters, padding and dynamic-range compression. This is not
interchangeable with the Matcha log-magnitude mel contract.
"""


def whisper_log_mel(waveforms, *, backend, n_mels=None):
    if n_mels is None:
        return backend.log_mel_spectrogram(waveforms)
    return backend.log_mel_spectrogram(waveforms, n_mels=n_mels)
