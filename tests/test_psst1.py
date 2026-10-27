import math
from pathlib import Path

import pytest
import torch

import quick_convert.systems.anonymization.psst1 as psst1_module
from quick_convert.components.speaker import SpeakerEmbedding
from quick_convert.components.speaker.emotion_compensation import EmotionBoundaryCompensator
from quick_convert.components.speaker.speaker_generators.psst1_flow import PSST1FlowSpeakerGenerator, mix_latents
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.external.psst1.flow import VelocityNet
from quick_convert.external.psst1.hifigan import Generator, upsample_condition
from quick_convert.systems.anonymization import PSST1Anonymizer


def write_flow(path: Path) -> Path:
    torch.manual_seed(0)
    torch.save(
        {"velocity_net": VelocityNet().state_dict(), "mean": torch.randn(192), "std": torch.rand(192) + 0.5},
        path,
    )
    return path


def write_boundaries(path: Path) -> Path:
    generator = torch.Generator().manual_seed(1)
    normals = {label: torch.randn(192, generator=generator) for label in EmotionBoundaryCompensator.EMOTIONS}
    torch.save({label: normal / normal.norm() for label, normal in normals.items()}, path)
    return path


def test_mix_latents_keeps_unit_variance_for_gaussian_inputs():
    generator = torch.Generator().manual_seed(0)
    z_orig = torch.randn(20_000, 192, generator=generator)
    z_rand = torch.randn(20_000, 192, generator=generator)

    mixed = mix_latents(z_orig, z_rand, 0.25)

    assert mixed.mean().abs() < 0.01
    assert (mixed.std() - 1.0).abs() < 0.01
    with pytest.raises(ValueError, match=r"w must be in \[0, 1\]"):
        mix_latents(z_orig, z_rand, 1.5)


def test_mix_latents_endpoints_select_one_input():
    z_orig, z_rand = torch.randn(3, 192), torch.randn(3, 192)

    assert torch.equal(mix_latents(z_orig, z_rand, 1.0), z_orig)
    assert torch.equal(mix_latents(z_orig, z_rand, 0.0), z_rand)


def test_flow_generator_standardizes_per_dimension(tmp_path):
    flow = PSST1FlowSpeakerGenerator(write_flow(tmp_path / "flow.pt"))
    x = torch.randn(4, 192)

    assert torch.allclose(flow.standardize(x), (x - flow.mean) / flow.std)
    assert torch.allclose(flow.destandardize(flow.standardize(x)), x, atol=1e-5)


def test_flow_generator_noise_follows_the_generator(tmp_path):
    flow = PSST1FlowSpeakerGenerator(write_flow(tmp_path / "flow.pt"))
    x = torch.randn(192)
    global_state = torch.random.get_rng_state()

    first = flow.anonymize(x, w=0.25, steps=3, generator=torch.Generator().manual_seed(5))
    second = flow.anonymize(x, w=0.25, steps=3, generator=torch.Generator().manual_seed(5))
    other = flow.anonymize(x, w=0.25, steps=3, generator=torch.Generator().manual_seed(6))

    assert first.shape == (192,)
    assert torch.equal(first, second)
    assert not torch.allclose(first, other)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert flow.anonymize(torch.randn(2, 192), w=0.25, steps=3).shape == (2, 192)


def test_flow_generator_preserves_caller_embeddings(tmp_path):
    flow = PSST1FlowSpeakerGenerator(write_flow(tmp_path / "flow.pt"))
    x = torch.randn(192)
    original = x.clone()

    flow.anonymize(x, w=0.25, steps=2)

    assert torch.equal(x, original)


@pytest.mark.parametrize(("emotion", "distance"), [("hap", 40.0), ("neu", 40.0), ("ang", 40.0), ("sad", -40.0)])
def test_emotion_compensation_sets_signed_boundary_distance(tmp_path, emotion, distance):
    compensator = EmotionBoundaryCompensator(write_boundaries(tmp_path / "boundaries.pt"))
    x = torch.randn(192)
    normal = compensator.normals[compensator.EMOTIONS.index(emotion)]

    out = compensator(x, emotion)

    scaled = x / x.norm() * math.sqrt(192)
    assert torch.allclose(out @ normal, torch.tensor(distance), atol=1e-4)
    # Only the component along the boundary normal changes.
    residual = out - scaled
    assert torch.allclose(residual - (residual @ normal) * normal, torch.zeros(192), atol=1e-4)


def test_emotion_compensation_rejects_unknown_labels(tmp_path):
    compensator = EmotionBoundaryCompensator(write_boundaries(tmp_path / "boundaries.pt"))

    with pytest.raises(ValueError, match="Unknown emotion 'surprise'"):
        compensator(torch.randn(192), "surprise")


def test_upsample_condition_repeats_frames():
    signal = torch.arange(6.0).reshape(1, 2, 3)

    out = upsample_condition(signal, 6)

    assert out.tolist() == [[[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]]]


def test_hifigan_generator_upsamples_f0_frames_to_samples():
    generator = Generator().eval()

    with torch.no_grad():
        out = generator(torch.randn(1, 393, 3))

    # 5 * 4 * 2 * 2 * 2 = 160 samples per 10 ms F0 frame.
    assert out.shape == (1, 1, 3 * 160)


class FakeSpeakerEncoder:
    def __init__(self, checkpoint_path, device=None):
        self.calls = 0

    def encode(self, wav, sr):
        self.calls += 1
        values = torch.linspace(-1.0, 1.0, 192) + wav.mean()
        return SpeakerEmbedding(values=values, embedding_dim=192, backend="fake", model_name="fake")


class FakeVocoder:
    SAMPLE_RATE = 16_000

    def __init__(self, hubert_path, generator_path, device=None):
        self.xvectors = []

    def __call__(self, waveform, xvector):
        self.xvectors.append(xvector.clone())
        return waveform.clone()


class FakeRecognizer:
    def __init__(self, label):
        self.label = label
        self.calls = 0

    def predict(self, waveform, sample_rate):
        self.calls += 1
        return self.label


@pytest.fixture
def make_psst1(tmp_path, monkeypatch):
    monkeypatch.setattr(psst1_module, "SpeechBrainECAPASpeakerEncoder", FakeSpeakerEncoder)
    monkeypatch.setattr(psst1_module, "PSST1Vocoder", FakeVocoder)
    write_flow(tmp_path / "flow.pt")
    write_boundaries(tmp_path / "emotion_boundaries.pt")
    for name in ("ecapa.ckpt", "hubert_soft.pt", "hifigan.pt"):
        (tmp_path / name).touch()

    def make(**kwargs):
        kwargs.setdefault("steps", 2)
        return PSST1Anonymizer(tmp_path, device="cpu", **kwargs)

    return make


def test_psst1_reports_missing_checkpoint_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="ecapa.ckpt"):
        PSST1Anonymizer(tmp_path, device="cpu")


def test_psst1_requires_an_emotion_source(make_psst1):
    anonymizer = make_psst1()

    with pytest.raises(ValueError, match="needs an emotion label"):
        anonymizer.anonymize(torch.randn(1600))


def test_psst1_explicit_emotion_wins_over_recognizer(make_psst1):
    recognizer = FakeRecognizer("sad")
    anonymizer = make_psst1(emotion_recognizer=recognizer, seed=0)
    waveform = torch.randn(1600)

    anonymizer.anonymize(waveform, emotion="hap")
    explicit = anonymizer.vocoder.xvectors[-1]
    anonymizer.anonymize(waveform)
    predicted = anonymizer.vocoder.xvectors[-1]

    assert recognizer.calls == 1
    hap = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("hap")]
    sad = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("sad")]
    assert torch.allclose(explicit @ hap, torch.tensor(40.0), atol=1e-3)
    assert torch.allclose(predicted @ sad, torch.tensor(-40.0), atol=1e-3)


def test_psst1_maps_recognizer_labels_and_rejects_unmapped_ones(make_psst1):
    anonymizer = make_psst1(emotion_recognizer=FakeRecognizer("happy"), emotion_label_map={"happy": "hap"})
    anonymizer.anonymize(torch.randn(1600))

    anonymizer.emotion_recognizer = FakeRecognizer("contempt")
    with pytest.raises(ValueError, match="'contempt' from the emotion_recognizer"):
        anonymizer.anonymize(torch.randn(1600))


def test_psst1_rejects_label_maps_outside_the_supported_emotions(make_psst1):
    with pytest.raises(ValueError, match="emotion_label_map targets"):
        make_psst1(emotion_label_map={"happy": "joy"})


def test_psst1_seed_depends_on_input_and_seed_only(make_psst1):
    anonymizer = make_psst1(seed=3)
    first, second = torch.randn(1600), torch.randn(1600)

    for waveform in (first, second, first):
        anonymizer.anonymize(waveform, emotion="neu")
    anonymizer.anonymize(first, emotion="neu", seed=4)
    xvectors = anonymizer.vocoder.xvectors

    assert torch.equal(xvectors[0], xvectors[2])
    assert not torch.allclose(xvectors[0], xvectors[1])
    assert not torch.allclose(xvectors[0], xvectors[3])


def test_psst1_without_seed_draws_fresh_noise(make_psst1):
    anonymizer = make_psst1()
    waveform = torch.randn(1600)

    anonymizer.anonymize(waveform, emotion="neu")
    anonymizer.anonymize(waveform, emotion="neu")

    assert not torch.allclose(*anonymizer.vocoder.xvectors)


def test_psst1_set_target_updates_defaults(make_psst1):
    anonymizer = make_psst1()

    anonymizer.set_target(w=0.5, steps=3)

    assert (anonymizer.w, anonymizer.steps) == (0.5, 3)
    with pytest.raises(ValueError, match="w must be in"):
        anonymizer.set_target(w=-0.1)


def test_psst1_batch_reads_emotion_resources(make_psst1):
    anonymizer = make_psst1(seed=0)

    def sample(name, label, length):
        return AudioSample(
            utt_id=name,
            path=Path(f"{name}.wav"),
            waveform=torch.randn(1, length),
            sample_rate=16_000,
            resources=ResourceCollection.from_refs([ResourceRef(name="emotion", kind="text", value=label)]),
        )

    batch = AudioBatch.from_samples([sample("a", "sad", 1600), sample("b", "ang", 1280)])
    generated = anonymizer.anonymize_batch(batch)

    sad = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("sad")]
    ang = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("ang")]
    assert generated.lengths.tolist() == [1600, 1280]
    assert torch.allclose(anonymizer.vocoder.xvectors[0] @ sad, torch.tensor(-40.0), atol=1e-3)
    assert torch.allclose(anonymizer.vocoder.xvectors[1] @ ang, torch.tensor(40.0), atol=1e-3)


def test_psst1_resynthesis_uses_the_original_xvector(make_psst1):
    anonymizer = make_psst1()
    waveform = torch.randn(1600)

    anonymizer.resynthesize(waveform)

    expected = FakeSpeakerEncoder(None).encode(waveform, 16_000).values
    assert torch.allclose(anonymizer.vocoder.xvectors[-1], expected)


def test_psst1_downloads_released_weights_when_no_directory_is_given(tmp_path, monkeypatch, make_psst1):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    PSST1Anonymizer(device="cpu", steps=2)

    assert calls == [
        {
            "repo_id": PSST1Anonymizer.PRETRAINED_REPO_ID,
            "revision": PSST1Anonymizer.PRETRAINED_REVISION,
            "allow_patterns": list(PSST1Anonymizer.CHECKPOINT_FILES.values()),
            "cache_dir": None,
        }
    ]


def test_psst1_download_accepts_another_repository(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path / kwargs["repo_id"]))

    assert PSST1Anonymizer.download("lab/psst1-variant", revision="abc123") == tmp_path / "lab/psst1-variant"
