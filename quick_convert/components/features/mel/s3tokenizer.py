"""S3Tokenizer mel recipe, delegated to the selected backend version.

The backend owns filters, padding and dynamic-range compression. This is not
interchangeable with the Matcha log-magnitude mel contract.
"""


def s3_log_mel(waveforms, *, backend, n_mels=128):
    return backend.log_mel_spectrogram(waveforms, n_mels=n_mels)
