"""Stage-by-stage parity between the PSST1 port and the research implementation.

References come from the research repository's export script (see
docs/components/psst1.md). Set ``QUICK_CONVERT_PSST1_EXPORT_DIR`` to a directory
holding ``assets/`` and ``references/`` to run these tests.
"""

import json
import os
from pathlib import Path

import pytest
import torch


EXPORT_DIR = os.environ.get("QUICK_CONVERT_PSST1_EXPORT_DIR")
pytestmark = pytest.mark.skipif(not EXPORT_DIR, reason="QUICK_CONVERT_PSST1_EXPORT_DIR is not set")
pytest.importorskip("speechbrain")
pytest.importorskip("amfm_decompy")

STEPS = 10


def references():
    if not EXPORT_DIR:
        return []
    manifest = json.loads((Path(EXPORT_DIR) / "references" / "manifest.json").read_text())
    return sorted(manifest)


@pytest.fixture(scope="module")
def assets():
    return Path(EXPORT_DIR) / "assets"


@pytest.fixture(scope="module")
def components(assets):
    from quick_convert.components.encoders.content.hubert_soft import SoftHubertContentEncoder
    from quick_convert.components.encoders.speaker.speechbrain_ecapa import SpeechBrainECAPASpeakerEncoder
    from quick_convert.components.features.f0.yaapt import YAAPTF0Extractor
    from quick_convert.components.generators.speaker.psst1_flow import PSST1FlowSpeakerGenerator
    from quick_convert.components.generators.waveform.psst1_hifigan import PSST1HifiGanDecoder
    from quick_convert.components.transforms.emotion_compensation import EmotionBoundaryCompensator

    return {
        "encoder": SpeechBrainECAPASpeakerEncoder(assets / "ecapa.ckpt", device="cpu"),
        "flow": PSST1FlowSpeakerGenerator(assets / "flow.pt"),
        "compensator": EmotionBoundaryCompensator(assets / "emotion_boundaries.pt"),
        "decoder": PSST1HifiGanDecoder(assets / "hifigan.pt", device="cpu"),
        "content": SoftHubertContentEncoder(assets / "hubert_soft.pt", device="cpu"),
        "f0": YAAPTF0Extractor(),
    }


def synthesize(components, waveform, xvector):
    from types import SimpleNamespace

    from quick_convert.systems.conversion.psst1 import PSST1ConversionSystem

    context = SimpleNamespace(
        content_encoder=components["content"],
        f0_extractor=components["f0"],
        decoder=components["decoder"],
        sample_rate=16000,
    )
    return PSST1ConversionSystem._synthesize(context, waveform, xvector)


def load_reference(utt):
    return torch.load(Path(EXPORT_DIR) / "references" / f"{utt}.pt", map_location="cpu", weights_only=False)


def snr_db(reference, estimate):
    noise = reference - estimate
    return float(10 * torch.log10(reference.pow(2).sum() / noise.pow(2).sum().clamp_min(1e-20)))


@pytest.mark.parametrize("utt", references())
def test_ecapa_matches_research(components, utt):
    ref = load_reference(utt)

    xvector = components["encoder"].encode(ref["wav"], 16_000).values

    assert torch.allclose(xvector, ref["xvector"], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("utt", references())
def test_flow_matches_research(components, utt):
    ref = load_reference(utt)
    flow = components["flow"]

    inverted = flow.invert(ref["flow_input"].unsqueeze(0), STEPS)[0]
    generated = flow.generate(ref["flow_z"].unsqueeze(0), STEPS)[0]

    assert torch.allclose(inverted, ref["flow_inverted"], atol=1e-4, rtol=1e-4)
    assert torch.allclose(generated, ref["flow_forward"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("utt", references())
def test_emotion_compensation_matches_research(components, utt):
    ref = load_reference(utt)

    out = components["compensator"](ref["emocomp_input"], ref["emotion"])

    assert torch.allclose(out, ref["emocomp_output"], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("utt", references())
def test_vocoder_features_match_research(components, utt):
    ref = load_reference(utt)
    encoder = components["content"]

    with torch.inference_mode():
        units = encoder.encode_waveforms(ref["vocoder_audio"].unsqueeze(0), sample_rate=16000).values[0]
    f0 = components["f0"].extract(ref["vocoder_audio"], 16_000)

    assert torch.allclose(units, ref["vocoder_soft_logits"], atol=1e-3, rtol=1e-3)
    # pYAAPT drifts by up to ~1 Hz across SciPy versions; voicing must agree.
    assert torch.equal(f0 > 0, ref["vocoder_f0"] > 0)
    assert torch.allclose(f0, ref["vocoder_f0"], atol=1.0)


@pytest.mark.parametrize("utt", references())
def test_vocoder_matches_research_given_research_f0(components, utt, monkeypatch):
    ref = load_reference(utt)
    monkeypatch.setattr(components["f0"], "extract", lambda *args, **kwargs: ref["vocoder_f0"].clone())

    wav = synthesize(components, ref["wav"], ref["emocomp_output"])

    assert wav.shape == ref["vocoder_wav"].shape
    assert snr_db(ref["vocoder_wav"], wav) > 50.0


@pytest.mark.parametrize("utt", references())
def test_vocoder_end_to_end_stays_close_to_research(components, utt):
    ref = load_reference(utt)

    wav = synthesize(components, ref["wav"], ref["emocomp_output"])

    # Sub-hertz F0 drift shifts the waveform phase, so this bound is loose.
    assert snr_db(ref["vocoder_wav"], wav) > 20.0
