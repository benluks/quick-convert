from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq_ema import ResidualVectorQuantizerEMA
from quick_convert.components.ssl import ContentFeatures, VQASRContentEncoder
from quick_convert.data import AudioBatch
from quick_convert.dependencies import dependency_plan
from quick_convert.inference import load_inference_system
from quick_convert.inference.source import CHECKPOINT_METADATA_KEY, inference_description
from quick_convert.systems.asr import VQASRSystem
from quick_convert.systems.conversion import SSLReconstructionConversionSystem
from quick_convert.utils.device import override_devices
from quick_convert.utils.masking import make_padding_mask
from quick_convert.utils.resolvers import register_config_resolvers


class TinyContentEncoder(nn.Module):
    device = torch.device("cpu")
    sample_rate = 16000
    frame_hz = 50

    @torch.inference_mode()
    def encode_waveforms(self, wavs, lengths=None):
        values = wavs[:, :12].reshape(-1, 3, 1, 4).expand(-1, -1, 2, -1)
        return ContentFeatures(values, lengths, 4, "continuous", "frame", "test", "tiny", None, 50)

    def forward(self, batch):
        return self.encode_waveforms(batch.waveforms, batch.lengths)

    def encode_file(self, path):
        return self.encode_waveforms(torch.ones(1, 12), torch.tensor([3]))


def make_source():
    return VQASRSystem(
        quantizer=ResidualVectorQuantizerEMA(
            input_dim=4, n_codebooks=1, codebook_size=4, codebook_dim=2, kmeans_init=False
        ),
        ctc_head=nn.Linear(4, 3),
        layer_fusion=LayerWeightedSum(2),
        online_encoders={"content": TinyContentEncoder()},
    ).eval()


def make_batch():
    return AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a"), Path("b")],
        splits=[None, None],
        resources={},
        waveforms=torch.randn(2, 12),
        lengths=torch.tensor([3, 2]),
        sample_rates=torch.tensor([16000, 16000]),
    )


def test_online_vq_matches_saved_bottleneck_and_decoder_backward_preserves_state():
    source = make_source()
    adapter = VQASRContentEncoder(source)
    batch = make_batch()
    with torch.no_grad():
        content = source.online_encoders["content"](batch)
        expected = source.quantizer(
            source.layer_fusion(content.values).transpose(1, 2),
            make_padding_mask(content.lengths, max_length=3),
        )
    decoder = nn.Linear(4, 1)
    system = SSLReconstructionConversionSystem(decoder=decoder, online_encoders={"content": adapter})
    system.train()
    adapter.train()
    assert all(not module.training for module in adapter.modules())
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert not any("ctc_head" in key for key in adapter.state_dict())
    before = {key: value.clone() for key, value in adapter.state_dict().items()}
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)
    decoder_before = decoder.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        resource = system.get_resource(batch, "content")
        torch.testing.assert_close(resource.values, expected.z_q.transpose(1, 2))
        assert resource.lengths.tolist() == [3, 2]
        assert not resource.values.is_inference()
        assert resource.values.shape[-1] != expected.latents.shape[1]
        decoder(resource.values).square().mean().backward()
        optimizer.step()
    assert not torch.equal(decoder.weight, decoder_before)
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in adapter.parameters())


def test_vq_content_waveforms_files_and_source_validation():
    adapter = VQASRContentEncoder(make_source())
    result = adapter.encode_waveforms(torch.ones(1, 12), torch.tensor([3]), sample_rate=16000)
    assert result.values.shape == (1, 3, 4)
    assert result.lengths.tolist() == [3]
    assert adapter.encode_file("unused.wav").values.shape == (1, 3, 4)
    with pytest.raises(ValueError, match="16000"):
        adapter.encode_waveforms(torch.ones(1, 12), torch.tensor([3]), sample_rate=8000)
    with pytest.raises(TypeError, match="VQASRSystem"):
        VQASRContentEncoder(nn.Identity())
    source = make_source()
    del source.online_encoders["content"]
    with pytest.raises(ValueError, match="online content encoder"):
        VQASRContentEncoder(source)


def test_vq_reconstruction_recipe_and_nested_loader_device(tmp_path):
    register_config_resolvers()
    # Dependency inspection needs only the saved composition, not model weights.
    OmegaConf.save(
        OmegaConf.create(
            {
                "system": {
                    "_target_": "quick_convert.systems.asr.VQASRSystem",
                    "online_encoders": {"content": {"_target_": "quick_convert.components.ssl.W2VBertContentEncoder"}},
                }
            }
        ),
        tmp_path / "config.yaml",
    )
    (tmp_path / "checkpoints").mkdir()
    torch.save({"state_dict": {"system.weight": torch.ones(1)}}, tmp_path / "checkpoints/last.ckpt")
    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="run/train_sslr_vq_asr_dit", overrides=[f"vq_asr_checkpoint={tmp_path}"])
    assert cfg.system.decoder.feature_dim == 1024
    assert cfg.system.decoder.flow._target_ == "quick_convert.external.cosyvoice.flow.flow.CausalMaskedDiffWithDiT"
    assert cfg.system.decoder.flow.decoder.estimator._target_ == "quick_convert.external.cosyvoice.flow.DiT.dit.DiT"
    assert "encoder" not in cfg.system.decoder.flow
    assert cfg.system.decoder.cond_strategy is None
    assert "feature_transform" not in cfg.system
    assert cfg.train_dataset.target_sr == 16000
    assert "content" not in cfg.train_dataset.load
    assert dependency_plan(cfg).extras == ("cosyvoice", "training", "transformers")
    resolved = OmegaConf.create(OmegaConf.to_container(cfg.system, resolve=True))
    override_devices(resolved, "cpu")
    assert resolved.online_encoders.content.system.map_location == "cpu"


def test_reconstruction_checkpoint_restores_nested_source_on_requested_device(tmp_path):
    source = make_source()
    source_path = tmp_path / "source.ckpt"
    source_config = {"system": {"_target_": "test_vq_asr_content.make_source"}}
    torch.save(
        {
            "state_dict": {f"system.{key}": value for key, value in source.state_dict().items()},
            CHECKPOINT_METADATA_KEY: inference_description(source_config),
        },
        source_path,
    )
    system = SSLReconstructionConversionSystem(
        decoder=nn.Linear(4, 1), online_encoders={"content": VQASRContentEncoder(source)}
    )
    config = {
        "system": {
            "_target_": "quick_convert.systems.conversion.SSLReconstructionConversionSystem",
            "decoder": {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 1},
            "online_encoders": {
                "content": {
                    "_target_": "quick_convert.components.ssl.VQASRContentEncoder",
                    "system": {
                        "_target_": "quick_convert.inference.load_inference_system",
                        "source": str(source_path),
                        "map_location": "cuda",
                    },
                }
            },
        }
    }
    checkpoint = tmp_path / "decoder.ckpt"
    torch.save(
        {
            "state_dict": {f"system.{key}": value for key, value in system.state_dict().items()},
            CHECKPOINT_METADATA_KEY: inference_description(config),
        },
        checkpoint,
    )
    restored = load_inference_system(checkpoint, map_location="cpu")
    batch = make_batch()
    torch.testing.assert_close(
        restored.get_resource(batch, "content").values, system.get_resource(batch, "content").values
    )
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
