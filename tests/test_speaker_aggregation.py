from pathlib import Path

import pytest
import torch

from quick_convert.components.speaker import SpeakerEmbeddingAggregation
from quick_convert.components.speaker.aggregation import audio_durations
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.resources import ResourceCollection, ResourceRef


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_shared_aggregation_weighting_and_input_ownership(dtype):
    embeddings = torch.tensor([[1.0, 10.0], [3.0, 30.0]], dtype=dtype)
    durations = torch.tensor([1.0, 3.0])
    before = embeddings.clone()
    mean = SpeakerEmbeddingAggregation()(embeddings, durations=durations)
    weighted = SpeakerEmbeddingAggregation("duration_weighted")(embeddings, durations=durations)
    torch.testing.assert_close(mean, torch.tensor([2.0, 20.0], dtype=dtype))
    torch.testing.assert_close(weighted, torch.tensor([2.5, 25.0], dtype=dtype))
    torch.testing.assert_close(embeddings, before)
    torch.testing.assert_close(durations, torch.tensor([1.0, 3.0]))
    assert weighted.dtype == dtype


@pytest.mark.parametrize(
    "values",
    [torch.ones(2, 3, 1), torch.empty(0, 3), torch.ones(2, 3, dtype=torch.long), torch.full((2, 3), float("nan"))],
)
def test_invalid_embeddings_are_rejected(values):
    with pytest.raises(ValueError):
        SpeakerEmbeddingAggregation()(values)


def test_concatenation_is_not_an_embedding_reduction():
    with pytest.raises(ValueError):
        SpeakerEmbeddingAggregation("concatenate")


def audio_batch():
    return AudioBatch.from_samples(
        [
            AudioSample(
                utt_id="short",
                path=Path("short.wav"),
                waveform=torch.ones(4),
                sample_rate=4,
                resources=ResourceCollection.from_refs([ResourceRef("spkid", "text", value="a")]),
            ),
            AudioSample(
                utt_id="long",
                path=Path("long.wav"),
                waveform=torch.ones(12),
                sample_rate=4,
                resources=ResourceCollection.from_refs([ResourceRef("spkid", "text", value="a")]),
            ),
            AudioSample(
                utt_id="other",
                path=Path("other.wav"),
                waveform=torch.ones(4),
                sample_rate=4,
                resources=ResourceCollection.from_refs([ResourceRef("spkid", "text", value="b")]),
            ),
        ]
    )


def test_valid_audio_duration_excludes_padding_and_uses_each_rate():
    batch = audio_batch()
    batch.sample_rates[2] = 2
    assert audio_durations(batch) == (1.0, 3.0, 2.0)
    assert batch.waveforms.shape == (3, 12)
    assert batch.lengths.tolist() == [4, 12, 4]
