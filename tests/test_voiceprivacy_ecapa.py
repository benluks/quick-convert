import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from quick_convert.components.speaker import VoicePrivacyECAPAEncoder
from quick_convert.data import AudioBatch


def test_voiceprivacy_config_composes(monkeypatch):
    monkeypatch.setenv("VOICEPRIVACY_ASV_DIR", "/models/asv_ssl")
    with initialize_config_dir(version_base=None, config_dir=str(Path("quick_convert/configs").resolve())):
        cfg = compose(config_name="run/eval_asv_voiceprivacy_ecapa")
    assert cfg.speaker_encoder._target_ == "quick_convert.components.speaker.VoicePrivacyECAPAEncoder"
    assert cfg.speaker_encoder.ecapa_checkpoint == "/models/asv_ssl/embedding_model.ckpt"


def test_voiceprivacy_batch_uses_true_lengths_and_stays_frozen():
    encoder = VoicePrivacyECAPAEncoder.__new__(VoicePrivacyECAPAEncoder)
    nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    seen = []

    def encode(wav, sr):
        seen.append((len(wav), sr))
        return SimpleNamespace(values=torch.ones(192))

    encoder.encode = encode
    encoder.train()
    assert not encoder.training
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a"), Path("b")],
        splits=[None, None],
        resources={},
        waveforms=torch.ones(2, 800),
        lengths=torch.tensor([800, 500]),
        sample_rates=torch.tensor([16000, 16000]),
    )
    assert encoder.encode_batch(batch).values.shape == (2, 192)
    assert seen == [(800, 16000), (500, 16000)]


@pytest.mark.skipif(not os.environ.get("VOICEPRIVACY_ASV_DIR"), reason="Requires official checkpoints")
def test_voiceprivacy_real_checkpoint_inference():
    root = Path(os.environ["VOICEPRIVACY_ASV_DIR"])
    encoder = VoicePrivacyECAPAEncoder(root / "embedding_model.ckpt", root / "WavLM-Large.pt", device="cpu")
    torch.manual_seed(115)
    wav = torch.randn(16000) * 0.1
    original = wav.clone()
    result = encoder.encode(wav, 16000)
    assert result.values.shape == (192,)
    assert torch.isfinite(result.values).all()
    torch.testing.assert_close(result.values.norm(), torch.tensor(1.0))
    torch.testing.assert_close(wav, original)
    assert not any(p.requires_grad for p in encoder.parameters())
    with pytest.raises(ValueError, match="non-silent"):
        encoder.encode(torch.zeros(16000), 16000)
    with pytest.raises(ValueError, match="16000 Hz"):
        encoder.encode(wav, 8000)

    # Optional parity check against the unmodified official ECAPA source.
    # Clone the pinned repository and set VOICEPRIVACY_SOURCE_DIR to its root.
    source = os.environ.get("VOICEPRIVACY_SOURCE_DIR")
    if source:
        import importlib.util

        import numpy as np
        from torch.nn import functional as F

        spec = importlib.util.spec_from_file_location(
            "official_ecapa", Path(source) / "evaluation/privacy/asv/ecapa_model_wavlm_24layer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.ECAPA_TDNN_test().eval()
        state = torch.load(root / "embedding_model.ckpt", map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        reference.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        with torch.inference_mode():
            # Same dual-input procedure as official SpeechBrainVectors.
            audio = torch.tensor(np.trim_zeros((wav / wav.abs().max()).numpy())).unsqueeze(0)
            fbank = encoder.normalizer(encoder.fbank(audio), torch.ones(1))
            ssl_audio = F.layer_norm(wav, wav.shape) if encoder.wavlm.cfg.normalize else wav
            outputs, _ = encoder.wavlm.extract_features(ssl_audio.unsqueeze(0), output_layer=24, ret_layer_results=True)
            layers = torch.stack([x for x, _ in outputs[1][1:]], dim=0).permute(2, 0, 1, 3)
            expected = F.normalize(reference(fbank, layers), dim=-1).squeeze()
        torch.testing.assert_close(result.values, expected, rtol=1e-5, atol=1e-6)
