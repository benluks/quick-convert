"""Migrate known saved Hydra targets without retaining old Python modules."""

from collections.abc import Mapping


_TARGETS = {
    "quick_convert.systems.reconstruction.SSLReconstructionSystem": "quick_convert.systems.conversion.ssl.SSLReconstructionConversionSystem",
    "quick_convert.systems.reconstruction.ssl.SSLReconstructionSystem": "quick_convert.systems.conversion.ssl.SSLReconstructionConversionSystem",
    "quick_convert.systems.anonymization.KNNVCAnonymizer": "quick_convert.systems.conversion.knnvc.KNNVCConversionSystem",
    "quick_convert.systems.anonymization.knnvc.KNNVCAnonymizer": "quick_convert.systems.conversion.knnvc.KNNVCConversionSystem",
    "quick_convert.pipelines.anonymization.KNNVCAnonymizer": "quick_convert.systems.conversion.knnvc.KNNVCConversionSystem",
    "quick_convert.pipelines.anonymization.knnvc.KNNVCAnonymizer": "quick_convert.systems.conversion.knnvc.KNNVCConversionSystem",
    "quick_convert.systems.anonymization.ASRBNAnonymizer": "quick_convert.systems.conversion.asrbn.ASRBNConversionSystem",
    "quick_convert.systems.anonymization.asrbn.ASRBNAnonymizer": "quick_convert.systems.conversion.asrbn.ASRBNConversionSystem",
    "quick_convert.pipelines.anonymization.ASRBNAnonymizer": "quick_convert.systems.conversion.asrbn.ASRBNConversionSystem",
    "quick_convert.pipelines.anonymization.asrbn.ASRBNAnonymizer": "quick_convert.systems.conversion.asrbn.ASRBNConversionSystem",
    "quick_convert.systems.anonymization.targets.ReferenceSpeech": "quick_convert.systems.conversion.targets.ReferenceSpeech",
    "quick_convert.systems.anonymization.targets.speech.ReferenceSpeech": "quick_convert.systems.conversion.targets.ReferenceSpeech",
    "quick_convert.systems.anonymization.targets.PreparedTarget": "quick_convert.systems.conversion.targets.PreparedTarget",
    "quick_convert.systems.anonymization.targets.speech.PreparedTarget": "quick_convert.systems.conversion.targets.PreparedTarget",
    "quick_convert.systems.reference.ReferenceCondition": "quick_convert.systems.conversion.reference.ReferenceCondition",
}


def migrate_system_config(value):
    """Copy a composition and rewrite only known class target fields."""
    if isinstance(value, Mapping):
        return {
            key: _TARGETS.get(item, item)
            if key == "_target_" and isinstance(item, str)
            else migrate_system_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [migrate_system_config(item) for item in value]
    return value
