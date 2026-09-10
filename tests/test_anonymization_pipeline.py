import pytest

from quick_convert.pipelines.anonymization.pipeline import AnonymizationPipeline


def test_anonymization_rejects_unimplemented_batching():
    with pytest.raises(NotImplementedError, match="variable-length output contract"):
        AnonymizationPipeline(
            anonymizer=object(),
            dataset=object(),
            batch_size=2,
        )
