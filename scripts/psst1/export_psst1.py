"""Export PSST1 assets and per-stage parity references from the research repo.

Run from $GEN with the research venv:  source env.sh && python <this file> <out_dir>
Reads the research repo only; writes everything under <out_dir>.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


GEN = Path.cwd()
EC = GEN.parent
assert (GEN / "env.sh").exists(), "run from the research gen/ directory"
sys.path[:0] = [
    str(GEN),
    str(GEN / "adapted_from_facebookresearch"),
    str(GEN / "selec_anon/compute_anon_spk_vector/flow_matching"),
]

OUT = Path(sys.argv[1])
ASSETS = OUT / "assets"
REFS = OUT / "references"
ASSETS.mkdir(parents=True, exist_ok=True)
REFS.mkdir(parents=True, exist_ok=True)

STEPS = 10
W = 0.25
UTTS = {
    "Ses01F_impro01_M011": "ang",
    "Ses01F_impro02_M001": "sad",
}
utt2emo = dict(line.split() for line in open(EC / "data/IEMOCAP_test/utt2emo"))
wav_scp = dict(line.split() for line in open(EC / "data/IEMOCAP_test/wav.scp"))
for label in ("hap", "neu"):
    UTTS[next(u for u, e in utt2emo.items() if e == label)] = label
report = {}


def maxdiff(a, b):
    return float((torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max())


# ---------------------------------------------------------------- HuBERT-soft
import fairseq  # noqa: E402
from torchaudio.models import hubert_base  # noqa: E402
from torchaudio.models.wav2vec2.utils import import_fairseq_model  # noqa: E402


(fs_model,), _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [str(GEN / "pretrained_models_anon_xv/hubert_base_ls960.pt")]
)
fs_model.remove_pretraining_modules()
soft = torch.load(GEN / "pretrained_models_anon_xv/HuBERT_soft/soft_model.pt", map_location="cpu")
fs_model.load_state_dict(
    {k[len("ssl_model.") :]: v for k, v in soft.items() if k.startswith("ssl_model.")}, strict=False
)
fs_model.eval()
ta_model = import_fairseq_model(fs_model).eval()
ta_sd = {k: v.clone() for k, v in ta_model.state_dict().items()}
reloaded = hubert_base()
reloaded.load_state_dict(ta_sd, strict=True)
reloaded.eval()
torch.save({"hubert": ta_sd, "W": soft["W"].clone()}, ASSETS / "hubert_soft.pt")

# ---------------------------------------------------------------- HiFi-GAN
from models_test import Generator, latentGenerator  # noqa: E402
from utils import AttrDict  # noqa: E402


hifi_dir = GEN / "pretrained_models_anon_xv/HiFi-GAN/libri_tts_clean_100_fbank_xv_ssl_freeze"
h = AttrDict(json.loads((hifi_dir / "config.json").read_text()))
g_sd = torch.load(hifi_dir / "g_00100000", map_location="cpu")["generator"]
bare = Generator(h)
bare.load_state_dict(
    {k: v for k, v in g_sd.items() if not k.startswith(("latent_encoder.", "xv_model.", "fbank."))}, strict=True
)
bare.eval()
bare.remove_weight_norm()
torch.save({"generator": bare.state_dict()}, ASSETS / "hifigan.pt")
report["hifigan_params"] = sum(v.numel() for v in bare.state_dict().values())

# ---------------------------------------------------------------- ECAPA
from speechbrain.lobes.features import Fbank  # noqa: E402
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN  # noqa: E402
from speechbrain.processing.features import InputNormalization  # noqa: E402


ecapa_ckpt = GEN / "pretrained_models_anon_xv/ECAPA-TDNN/embedding_model.ckpt"
shutil.copy(ecapa_ckpt, ASSETS / "ecapa.ckpt")
ecapa = ECAPA_TDNN(input_size=80, lin_neurons=192)
missing = ecapa.load_state_dict(torch.load(ecapa_ckpt, map_location="cpu"), strict=True)
ecapa.eval()
fbank = Fbank(n_mels=80)
mvn = InputNormalization(norm_type="sentence", std_norm=False)

# ---------------------------------------------------------------- flow
from embeddings_generator import EmbeddingsGenerator  # noqa: E402


flow_ck = torch.load(
    GEN / "trained/flow_models/flow_301_epochs_enrolls/flow.pt", map_location="cpu", weights_only=False
)
torch.save(
    {
        "velocity_net": flow_ck["model_state_dict"],
        "mean": torch.as_tensor(flow_ck["mean"], dtype=torch.float32),
        "std": torch.as_tensor(flow_ck["std"], dtype=torch.float32),
    },
    ASSETS / "flow.pt",
)
egen = EmbeddingsGenerator(str(GEN / "trained/flow_models/flow_301_epochs_enrolls/flow.pt"), device="cpu")
fm = egen.flow_matcher
flow_mean = torch.as_tensor(flow_ck["mean"], dtype=torch.float32)
flow_std = torch.as_tensor(flow_ck["std"], dtype=torch.float32)


# ---------------------------------------------------------------- boundaries
def linear_interpolate(latent_code, boundary, start_distance=-3.0, end_distance=3.0, steps=10):
    """Verbatim 2-D branch of $EC/utils/manipulator.py (that module has package-relative imports)."""
    assert latent_code.shape[0] == 1 and boundary.shape[0] == 1 and len(boundary.shape) == 2
    linspace = np.linspace(start_distance, end_distance, steps)
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary


long_names = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad"}
boundaries = {
    short: torch.from_numpy(np.load(EC / f"boundaries_esd_msp/ESD_MSP_emo_inter_{name}/boundary.npy"))
    .float()
    .reshape(192)
    for short, name in long_names.items()
}
report["boundary_norms"] = {k: float(v.norm()) for k, v in boundaries.items()}
torch.save(boundaries, ASSETS / "emotion_boundaries.pt")


def research_emocomp(vec, emo):
    """edit_spk_emo_pre_esd_msp_sad1.py with norm_type=inter, start=30, end=40, steps=2."""
    code = vec.reshape(1, 192).astype(np.float64)
    code = code / np.linalg.norm(code, axis=1, keepdims=True) * np.sqrt(192)
    boundary = np.load(EC / f"boundaries_esd_msp/ESD_MSP_emo_inter_{long_names[emo]}/boundary.npy")
    start, end = (-30.0, -40.0) if emo == "sad" else (30.0, 40.0)
    return linear_interpolate(code, boundary, start_distance=start, end_distance=end, steps=2)[1:]


# ---------------------------------------------------------------- vocoder (research path, fairseq)
from dataset_test import latentDataset  # noqa: E402
from kaldiio import ReadHelper  # noqa: E402
from scripts.readwrite import read_raw_mat  # noqa: E402,F401


os.chdir(GEN)
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "map_location": "cpu"})  # research code assumes CUDA
vocoder = latentGenerator(h)
torch.load = _torch_load
vocoder.load_state_dict(g_sd)
vocoder.eval()
vocoder.remove_weight_norm()

stored_xv = {}
with ReadHelper(f"scp:{EC / 'compute_ori_spk_vector/output_ori_spk_vector/IEMOCAP_test/xvector.scp'}") as reader:
    for key, mat in reader:
        if key in UTTS:
            stored_xv[key] = mat

# ---------------------------------------------------------------- per-utterance references
manifest = {}
gen = torch.Generator().manual_seed(1234)
for utt, emo in UTTS.items():
    wav_path = EC / wav_scp[utt]
    wav, sr = torchaudio.load(wav_path)
    assert sr == 16000, sr
    ref = {"wav_path": str(wav_path), "emotion": emo, "wav": wav[0].clone()}

    with torch.no_grad():
        # ECAPA x-vector (extract_emb.compute_embedding with the checkpoint actually loaded)
        feats = mvn(fbank(wav), torch.ones(1))
        xv = ecapa(feats, torch.ones(1)).reshape(192)
        ref["xvector"] = xv

        # HuBERT conversion check on real audio
        fs_x = fs_model(wav, mask=False, features_only=True)["x"]
        ta_x, _ = reloaded(wav)
        report.setdefault("hubert_fairseq_vs_torchaudio_maxdiff", {})[utt] = maxdiff(fs_x, ta_x)

        # flow network on per-dimension normalized input (independent of the research normalization)
        x_norm = ((xv - flow_mean) / flow_std).unsqueeze(0)
        ref["flow_input"] = x_norm[0]
        ref["flow_inverted"] = fm.invert_embedding(x_norm, steps=STEPS)[0]
        z = torch.randn(1, 192, generator=gen)
        ref["flow_z"] = z[0]
        ref["flow_forward"] = fm.embedding(z, steps=STEPS)[0]

        # research normalization, for the record (EmbeddingsGenerator._normalize)
        ref["research_normalized"] = egen._normalize(xv.unsqueeze(0), flow_ck["mean"], flow_ck["std"])[0]

        # emotion compensation on a fixed vector
        emo_in = torch.randn(192, generator=gen) * 0.3 + flow_mean
        ref["emocomp_input"] = emo_in
        ref["emocomp_output"] = torch.from_numpy(research_emocomp(emo_in.numpy(), emo)).float().reshape(192)

        # vocoder: research preprocessing + gen_vpc with the emocomp vector
        ds = latentDataset(
            [str(wav_path)],
            -1,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            n_cache_reuse=0,
            fmax_loss=h.fmax_for_loss,
            device="cpu",
        )
        feats_v, audio_pre, _, _ = ds[0]
        ref["vocoder_audio"] = feats_v["audio"][0, 0] if feats_v["audio"].dim() == 3 else feats_v["audio"][0]
        ref["vocoder_f0"] = feats_v["f0"].reshape(-1)
        ref["vocoder_soft_logits"] = vocoder.latent_encoder(
            feats_v["audio"].unsqueeze(0) if feats_v["audio"].dim() == 2 else feats_v["audio"]
        )[0]
        xv_voc = ref["emocomp_output"].numpy().astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".xvector", delete=False) as tmp:
            xv_voc.tofile(tmp.name)
        x_in = {k: v.unsqueeze(0) if k == "audio" and v.dim() == 2 else v for k, v in feats_v.items()}
        y = vocoder.gen_vpc(tmp.name, **x_in)
        os.remove(tmp.name)
        ref["vocoder_raw"] = y.reshape(-1)
        audio = (y.squeeze() * 32768.0).numpy().astype("int16")[: wav.shape[-1]]
        import librosa

        ref["vocoder_wav"] = torch.from_numpy(librosa.util.normalize(audio.astype(np.float32)))

    if utt in stored_xv:
        report.setdefault("stored_vs_loaded_ecapa_cosine", {})[utt] = float(
            F.cosine_similarity(torch.as_tensor(stored_xv[utt]).reshape(1, -1), xv.reshape(1, -1))
        )
    torch.save(ref, REFS / f"{utt}.pt")
    manifest[utt] = {"emotion": emo, "wav_path": str(wav_path), "num_samples": int(wav.shape[-1])}

(REFS / "manifest.json").write_text(json.dumps(manifest, indent=2))

# sanity: are flow mean/std close to real ECAPA statistics?
xs = torch.stack([torch.load(REFS / f"{u}.pt")["xvector"] for u in UTTS])
report["flow_mean_vs_xvector_mean_corr"] = float(np.corrcoef(xs.mean(0).numpy(), flow_mean.numpy())[0, 1])
report["xvector_abs_mean"] = float(xs.abs().mean())
report["flow_mean_abs_mean"] = float(flow_mean.abs().mean())
(OUT / "export_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
