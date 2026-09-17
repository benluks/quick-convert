from pathlib import Path

import torch

from quick_convert.data import AudioBatch
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.training.lightning.modules import base as base_module
from quick_convert.training.lightning.modules.base import BaseTrainingModule


def test_batch_transfer_moves_model_inputs_without_traversing_resource_refs(monkeypatch) -> None:
    resource_refs = [
        ResourceCollection.from_refs(
            [
                ResourceRef(
                    name="token_ids",
                    kind="token_ids",
                    value=torch.tensor([1, 2]),
                )
            ]
        )
    ]
    batch = AudioBatch(
        utt_ids=["utterance"],
        paths=[Path("utterance.flac")],
        splits=["train"],
        resources={"token_ids": torch.tensor([[1, 2]])},
        resource_refs=resource_refs,
        waveforms=torch.ones(1, 4),
        lengths=torch.tensor([4]),
        sample_rates=torch.tensor([16_000]),
    )
    transferred = []

    def record_transfer(value, device):
        transferred.append(value)
        return value

    monkeypatch.setattr(base_module, "move_data_to_device", record_transfer)

    result = BaseTrainingModule.transfer_batch_to_device(
        object(),
        batch,
        torch.device("cuda"),
        dataloader_idx=0,
    )

    assert result is batch
    assert len(transferred) == 4
    assert transferred[0] is batch.waveforms
    assert transferred[1] is batch.lengths
    assert transferred[2] is batch.sample_rates
    assert transferred[3] is batch.resources
    assert batch.resource_refs is resource_refs
