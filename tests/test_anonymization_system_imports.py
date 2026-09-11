from quick_convert.pipelines.anonymization import (
    ASRBNAnonymizer as LegacyASRBNAnonymizer,
)
from quick_convert.pipelines.anonymization import (
    BaseAnonymizer as LegacyBaseAnonymizer,
)
from quick_convert.pipelines.anonymization import (
    KNNVCAnonymizer as LegacyKNNVCAnonymizer,
)
from quick_convert.systems.anonymization import ASRBNAnonymizer, BaseAnonymizer, KNNVCAnonymizer


def test_pipeline_imports_reexport_anonymization_systems() -> None:
    assert LegacyBaseAnonymizer is BaseAnonymizer
    assert LegacyASRBNAnonymizer is ASRBNAnonymizer
    assert LegacyKNNVCAnonymizer is KNNVCAnonymizer
