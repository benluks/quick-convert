import math
from pathlib import Path

import pytest
import torch

import quick_convert.systems.conversion.psst1 as psst1_module
from quick_convert.components.encoders.speaker import SpeakerEmbedding
from quick_convert.components.flow_matching.estimators.psst1 import PSST1FlowEstimator
from quick_convert.components.generators.speaker.psst1_flow import PSST1FlowSpeakerGenerator, mix_latents
from quick_convert.components.generators.waveform.psst1_hifigan import PSST1HifiGanGenerator, upsample_condition
from quick_convert.components.transforms.emotion_compensation import EmotionBoundaryCompensator
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.data.resources import ResourceCollection, ResourceRef
from quick_convert.systems.conversion import GeneratedVoice, PSST1ConversionSystem


def write_flow(path: Path) -> Path:
    torch.manual_seed(0)
    torch.save(
        {"velocity_net": PSST1FlowEstimator().state_dict(), "mean": torch.randn(192), "std": torch.rand(192) + 0.5},
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
    generator = PSST1HifiGanGenerator().eval()

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


class FakeDecoder:
    SAMPLE_RATE = 16_000

    def __init__(self, generator_path, device=None):
        self.xvectors = []

    def __call__(self, units, f0, xvector, *, num_samples=None):
        self.xvectors.append(xvector.clone())
        return units[:, 0].clone()[:num_samples]


class FakeContentEncoder(torch.nn.Module):
    def __init__(self, checkpoint_path, device=None):
        super().__init__()

    def encode_waveforms(self, waveforms, sample_rate):
        from types import SimpleNamespace

        return SimpleNamespace(values=waveforms[..., None], lengths=torch.tensor([waveforms.shape[1]]))


class FakeF0Extractor:
    def extract(self, audio, sample_rate):
        return torch.zeros(max(1, len(audio) // 160))


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
    monkeypatch.setattr(psst1_module, "PSST1HifiGanDecoder", FakeDecoder)
    monkeypatch.setattr(psst1_module, "SoftHubertContentEncoder", FakeContentEncoder)
    monkeypatch.setattr(psst1_module, "YAAPTF0Extractor", FakeF0Extractor)
    write_flow(tmp_path / "flow.pt")
    write_boundaries(tmp_path / "emotion_boundaries.pt")
    for name in ("ecapa.ckpt", "hubert_soft.pt", "hifigan.pt"):
        (tmp_path / name).touch()

    def make(**kwargs):
        kwargs.setdefault("steps", 2)
        return PSST1ConversionSystem(tmp_path, device="cpu", **kwargs)

    return make


def test_psst1_reports_missing_checkpoint_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="ecapa.ckpt"):
        PSST1ConversionSystem(tmp_path, device="cpu")


def test_psst1_requires_an_emotion_source(make_psst1):
    anonymizer = make_psst1()

    with pytest.raises(ValueError, match="needs an emotion label"):
        anonymizer.convert(torch.randn(1600), target=anonymizer.prepare_target(GeneratedVoice()))


def test_psst1_explicit_emotion_wins_over_recognizer(make_psst1):
    recognizer = FakeRecognizer("sad")
    anonymizer = make_psst1(emotion_recognizer=recognizer)
    waveform = torch.randn(1600)

    anonymizer.convert(waveform, target=anonymizer.prepare_target(GeneratedVoice()), emotion="hap")
    explicit = anonymizer.decoder.xvectors[-1]
    anonymizer.convert(waveform, target=anonymizer.prepare_target(GeneratedVoice()))
    predicted = anonymizer.decoder.xvectors[-1]

    assert recognizer.calls == 1
    hap = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("hap")]
    sad = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("sad")]
    assert torch.allclose(explicit @ hap, torch.tensor(40.0), atol=1e-3)
    assert torch.allclose(predicted @ sad, torch.tensor(-40.0), atol=1e-3)


def test_psst1_maps_recognizer_labels_and_rejects_unmapped_ones(make_psst1):
    anonymizer = make_psst1(emotion_recognizer=FakeRecognizer("happy"), emotion_label_map={"happy": "hap"})
    anonymizer.convert(torch.randn(1600), target=anonymizer.prepare_target(GeneratedVoice()))

    anonymizer.emotion_recognizer = FakeRecognizer("contempt")
    with pytest.raises(ValueError, match="'contempt' from the emotion_recognizer"):
        anonymizer.convert(torch.randn(1600), target=anonymizer.prepare_target(GeneratedVoice()))


def test_psst1_rejects_label_maps_outside_the_supported_emotions(make_psst1):
    with pytest.raises(ValueError, match="emotion_label_map targets"):
        make_psst1(emotion_label_map={"happy": "joy"})


def test_psst1_seed_depends_on_input_and_seed_only(make_psst1):
    anonymizer = make_psst1()
    first, second = torch.randn(1600), torch.randn(1600)

    for waveform in (first, second, first):
        anonymizer.convert(waveform, target=anonymizer.prepare_target(GeneratedVoice(seed=3)), emotion="neu")
    anonymizer.convert(first, target=anonymizer.prepare_target(GeneratedVoice(seed=4)), emotion="neu")
    xvectors = anonymizer.decoder.xvectors

    assert torch.equal(xvectors[0], xvectors[2])
    assert not torch.allclose(xvectors[0], xvectors[1])
    assert not torch.allclose(xvectors[0], xvectors[3])


def test_psst1_validates_prepared_target_ownership(make_psst1):
    system = make_psst1()
    other = make_psst1()
    with pytest.raises(ValueError, match="incompatible"):
        other.convert(torch.randn(1600), target=system.prepare_target(GeneratedVoice()), emotion="neu")
    with pytest.raises(TypeError, match="PreparedTarget"):
        system.convert(torch.randn(1600), target=GeneratedVoice(), emotion="neu")
    with pytest.raises(ValueError, match="w must be"):
        make_psst1(w=-0.1)


def test_psst1_batch_reads_emotion_resources(make_psst1):
    anonymizer = make_psst1()

    def sample(name, label, length):
        return AudioSample(
            utt_id=name,
            path=Path(f"{name}.wav"),
            waveform=torch.randn(1, length),
            sample_rate=16_000,
            resources=ResourceCollection.from_refs([ResourceRef(name="emotion", kind="text", value=label)]),
        )

    batch = AudioBatch.from_samples([sample("a", "sad", 1600), sample("b", "ang", 1280)])
    generated = anonymizer.convert_batch(batch, target=anonymizer.prepare_target(GeneratedVoice()))

    sad = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("sad")]
    ang = anonymizer.emotion_compensator.normals[anonymizer.EMOTIONS.index("ang")]
    assert generated.lengths.tolist() == [1600, 1280]
    assert torch.allclose(anonymizer.decoder.xvectors[0] @ sad, torch.tensor(-40.0), atol=1e-3)
    assert torch.allclose(anonymizer.decoder.xvectors[1] @ ang, torch.tensor(40.0), atol=1e-3)


def test_psst1_resynthesis_uses_the_original_xvector(make_psst1):
    anonymizer = make_psst1()
    waveform = torch.randn(1600)

    anonymizer.resynthesize(waveform)

    expected = FakeSpeakerEncoder(None).encode(waveform, 16_000).values
    assert torch.allclose(anonymizer.decoder.xvectors[-1], expected)


def test_psst1_downloads_released_weights_when_no_directory_is_given(tmp_path, monkeypatch, make_psst1):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    PSST1ConversionSystem(device="cpu", steps=2)

    assert calls == [
        {
            "repo_id": PSST1ConversionSystem.PRETRAINED_REPO_ID,
            "revision": PSST1ConversionSystem.PRETRAINED_REVISION,
            "allow_patterns": list(PSST1ConversionSystem.CHECKPOINT_FILES.values()),
            "cache_dir": None,
        }
    ]


def test_psst1_download_accepts_another_repository(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path / kwargs["repo_id"]))

    assert PSST1ConversionSystem.download("lab/psst1-variant", revision="abc123") == tmp_path / "lab/psst1-variant"


def test_psst1_conversion_pipeline_writes_and_resumes(tmp_path, make_psst1):
    import json

    import soundfile as sf

    from quick_convert.data import BaseDataset
    from quick_convert.pipelines.conversion import SpeechConversionPipeline

    system = make_psst1()
    audio = tmp_path / "source.wav"
    sf.write(audio, torch.sin(torch.arange(1600) * 0.1).numpy(), 16000)
    sample = AudioSample(
        utt_id="utt",
        path=audio,
        split="test",
        resources=ResourceCollection.from_refs([ResourceRef("emotion", "text", value="neu")]),
    )
    pipeline = SpeechConversionPipeline(
        system, BaseDataset(rows=[sample], load=["audio"]), GeneratedVoice(52), tmp_path / "converted"
    )
    pipeline.run()
    output = pipeline.out_dir / "test/utt.wav"
    values, rate = sf.read(output)
    assert rate == 16000 and len(values) == 1600
    assert abs(values).max() > 0
    assert len(system.decoder.xvectors) == 1
    pipeline.run()
    assert len(system.decoder.xvectors) == 1
    assert json.loads((pipeline.out_dir / "conversions.jsonl").read_text())["utt_id"] == "utt"


def test_psst1_children_follow_parent_device_move():
    from quick_convert.components.encoders.speaker.speechbrain_ecapa import SpeechBrainECAPASpeakerEncoder
    from quick_convert.components.generators.waveform.psst1_hifigan import PSST1HifiGanDecoder
    from quick_convert.systems.ser.speechbrain_ser import SpeechBrainSER

    parent = torch.nn.Module()
    # Construct device-tracking shells without downloading optional models.
    for name, cls in (("vocoder", PSST1HifiGanDecoder), ("speaker", SpeechBrainECAPASpeakerEncoder)):
        child = cls.__new__(cls)
        torch.nn.Module.__init__(child)
        child.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        parent.add_module(name, child)
    parent.add_module("ser", SpeechBrainSER(device="cpu"))
    parent.to("meta")
    assert all(child.device.type == "meta" for child in parent.children())


def test_psst1_cli_composition_uses_conversion_pipeline(tmp_path, make_psst1, monkeypatch):
    import soundfile as sf
    from hydra import compose, initialize_config_dir

    from quick_convert.cli.run import execute_pipeline

    audio_dir = tmp_path / "audio" / "test" / "sentences" / "wav" / "session"
    audio_dir.mkdir(parents=True)
    sf.write(audio_dir / "source.wav", torch.sin(torch.arange(1600) * 0.1).numpy(), 16000)
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "quick_convert/configs")):
        cfg = compose(
            config_name="run/convert_psst1_iemocap",
            overrides=[
                f"dataset.root={tmp_path}/audio",
                "dataset.splits=[test]",
                "+dataset.resource_providers=[{_target_:quick_convert.data.resources.TemplateResourceProvider,name:emotion,template:neu}]",
                f"+system.checkpoint_dir={tmp_path}",
                "system.emotion_recognizer=null",
                "device=cpu",
                f"pipeline.out_dir={tmp_path}/cli-output",
            ],
        )
    # The fixture supplies local flow/boundary files and backend stand-ins.
    make_psst1()
    output_dir = execute_pipeline(cfg, print_config=False)
    import json

    row = json.loads((output_dir / "conversions.jsonl").read_text())
    assert row["assignment"]["mode"] == "generated"
    assert row["assignment"]["seed"] == 52
    assert row["assignment"]["target_speaker"] is None
    output = output_dir / "test/source.wav"
    assert output.is_file()
    previous = output.read_bytes()

    def unexpected_encode(*args, **kwargs):
        raise AssertionError("A resumed CLI run re-encoded a completed recording.")

    monkeypatch.setattr(FakeSpeakerEncoder, "encode", unexpected_encode)
    execute_pipeline(cfg, print_config=False)
    assert output.read_bytes() == previous
    cfg.target.seed = 53
    with pytest.raises(ValueError, match="target assignment differs"):
        execute_pipeline(cfg, print_config=False)


def test_lazy_ser_backend_moves_without_changing_checkpoint_keys():
    from quick_convert.systems.ser.speechbrain_ser import SpeechBrainSER

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def classify_batch(self, audio):
            assert self.weight.device == audio.device == self.device
            return None, None, None, ["NEU"]

    recognizer = SpeechBrainSER(device="cpu")
    before = set(recognizer.state_dict())
    # Simulate a backend loaded by foreign_class without downloading it.
    object.__setattr__(recognizer, "_classifier", Classifier())
    assert set(recognizer.state_dict()) == before
    parent = torch.nn.ModuleDict({"recognizer": recognizer}).to("meta")
    assert parent["recognizer"].predict(torch.ones(1600), 16000) == "neu"


def test_shared_hifigan_loads_folded_knn_weights_with_identical_output():
    from quick_convert.components.generators.waveform.knnvc_hifigan import Generator, KnnVCHifiGanConfig

    config = KnnVCHifiGanConfig(
        hubert_dim=4,
        hifi_dim=4,
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        upsample_initial_channel=16,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    knn = Generator(config).eval()
    with torch.no_grad():
        knn.lin_pre.weight.copy_(torch.eye(4))
        knn.lin_pre.bias.zero_()
    features = torch.randn(2, 3, 4)
    normalized = knn(features)
    knn.remove_weight_norm()

    psst = PSST1HifiGanGenerator(
        model_in_dim=4,
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        upsample_initial_channel=16,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    ).eval()
    folded_state = {key: value for key, value in knn.state_dict().items() if not key.startswith("lin_pre.")}
    psst.load_state_dict(folded_state, strict=True)
    torch.testing.assert_close(knn(features), normalized, rtol=0, atol=0)
    torch.testing.assert_close(psst(features.transpose(1, 2)), normalized, rtol=0, atol=0)


def test_split_resynthesis_matches_previous_waveform_path(tmp_path, monkeypatch):
    from functools import partial
    from types import SimpleNamespace

    import torch.nn.functional as F
    import torchaudio.models

    import quick_convert.components.generators.waveform.psst1_hifigan as hifigan
    from quick_convert.components.encoders.content.hubert_soft import SoftHubertContentEncoder

    class Hubert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scales", torch.arange(1, 5).float())

        def forward(self, audio):
            frames = audio.unfold(-1, 400, 320).mean(-1)
            return frames[..., None] * self.scales, None

    monkeypatch.setattr(torchaudio.models, "hubert_base", Hubert)
    generator = partial(
        PSST1HifiGanGenerator,
        upsample_initial_channel=32,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    torch.save({"hubert": Hubert().state_dict(), "W": torch.randn(200, 4)}, tmp_path / "hubert.pt")
    torch.save({"generator": generator().state_dict()}, tmp_path / "hifigan.pt")
    monkeypatch.setattr(hifigan, "PSST1HifiGanGenerator", generator)
    encoder = SoftHubertContentEncoder(tmp_path / "hubert.pt", device="cpu")
    decoder = hifigan.PSST1HifiGanDecoder(tmp_path / "hifigan.pt", device="cpu")

    class F0:
        def extract(self, audio, rate):
            return torch.linspace(100, 140, len(audio) // 160)

    context = SimpleNamespace(content_encoder=encoder, f0_extractor=F0(), decoder=decoder, sample_rate=16000)
    # Retain the previous waveform-to-conditioning order as a regression oracle.
    waveform, speaker = torch.randn(1607), torch.randn(192)
    audio = waveform.float().cpu()
    audio = (audio / audio.abs().max() * 0.95)[:1600]
    raw, _ = encoder.hubert(audio.unsqueeze(0))
    units = F.linear(F.normalize(raw), F.normalize(encoder.unit_projection))
    units = F.pad(F.layer_norm(units, units.shape).transpose(2, 1), (0, 1), "replicate")
    f0 = context.f0_extractor.extract(audio, 16000).reshape(1, 1, -1)
    units = upsample_condition(units, f0.shape[-1])
    vector = speaker.reshape(1, 192, 1)
    vector = F.layer_norm(vector, vector.shape)
    packed = torch.cat((units, f0, upsample_condition(vector, f0.shape[-1])), dim=1)
    with torch.inference_mode():
        expected = decoder.generator(packed).reshape(-1)[: len(waveform)]
        expected = expected / expected.abs().max()
        actual = PSST1ConversionSystem._synthesize(context, waveform, speaker)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # The content contract also preserves valid lengths when batching recordings.
    batch = torch.stack((F.pad(audio[:960], (0, 640)), audio))
    features = encoder.encode_waveforms(batch, torch.tensor([960, 1600]), 16000)
    alone = encoder.encode_waveforms(audio[:960].unsqueeze(0), sample_rate=16000)
    assert features.lengths.tolist() == [2, 4]
    torch.testing.assert_close(features.values[0, :2], alone.values[0], rtol=0, atol=0)
