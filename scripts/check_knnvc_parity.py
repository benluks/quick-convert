"""Compare modular kNN-VC with the upstream pretrained implementation."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from quick_convert.components.conditioning.nearest_neighbour import CosineTopK, NearestNeighbourMatching
from quick_convert.components.conditioning.nearest_neighbour_reference import NearestNeighbourReferenceConditioning
from quick_convert.components.encoders.content.original_wavlm import OriginalWavLMContentEncoder
from quick_convert.components.generators.waveform.knnvc_hifigan import KnnVCHifiGanDecoder
from quick_convert.systems.conversion.ssl_reconstruction import SSLReconstructionConversionSystem
from quick_convert.systems.conversion.targets import ReferenceSpeech
from quick_convert.utils.audio import load_audio_input


def compare(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    result = {"upstream_shape": list(a.shape), "modular_shape": list(b.shape)}
    if a.shape == b.shape:
        delta = a - b
        result.update(mae=delta.abs().mean().item(), max_abs=delta.abs().max().item())
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("reference", type=Path, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/knnvc_parity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    upstream = torch.hub.load(
        "bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=args.device
    ).eval()
    query = upstream.get_features(str(args.source))
    pool = upstream.get_matching_set([str(path) for path in args.reference], vad_trigger_level=0).to(query.device)
    # Reproduce upstream's actual distance arithmetic, not just its mathematical equivalent.
    from matcher import fast_cosine_dist

    indices = fast_cosine_dist(query, pool, device=args.device).topk(4, largest=False).indices
    matched = pool[indices].mean(dim=1)
    original = upstream.match(query, pool, topk=4, tgt_loudness_db=None)
    selector = CosineTopK()
    matcher = NearestNeighbourMatching(selector)
    ours_indices = selector(query, pool)
    ours_matched = matcher(query, pool)
    report = {
        "upstream_input_neighbour_set_agreement": (indices.sort(dim=-1).values == ours_indices.sort(dim=-1).values)
        .all(dim=-1)
        .float()
        .mean()
        .item(),
        "upstream_input_aggregation": compare(matched, ours_matched),
    }
    decoder = KnnVCHifiGanDecoder.from_pretrained(device=args.device).eval()
    report["decoder_same_input"] = compare(upstream.vocode(matched[None]), decoder(matched[None]))
    del upstream
    encoder = OriginalWavLMContentEncoder(device=args.device)
    system = SSLReconstructionConversionSystem(
        decoder=decoder,
        online_encoders={"content": encoder},
        reference_conditioning=NearestNeighbourReferenceConditioning(matcher),
    ).eval()
    waveform = load_audio_input(args.source, target_sample_rate=16000, mono=True).reshape(-1)
    batch = system._reference_batch(waveform, 16000)
    content = system.get_resource(batch, "content")
    features, lengths, _ = system.encode_features(content.values, content.lengths, metadata=content.metadata)
    target = system.prepare_target(ReferenceSpeech(args.reference))
    modular_query = features[0, : int(lengths[0])]
    report["source_features"] = compare(query, modular_query)
    modular_indices = selector(modular_query, target.conditioning)
    report["modular_neighbour_set_agreement"] = (
        (indices.sort(dim=-1).values == modular_indices.sort(dim=-1).values).all(dim=-1).float().mean().item()
        if indices.shape == modular_indices.shape
        else None
    )
    report["reference_pool"] = compare(pool, target.conditioning)
    report["modular_aggregation"] = compare(matched, matcher(modular_query, target.conditioning))
    converted = system.convert_batch(batch, target=target).waveform(0)
    report["end_to_end"] = compare(original.reshape(-1), converted.reshape(-1))
    report["settings"] = {"reference_vad": False, "output_loudness_normalization": False, "k": 4}
    for name, audio in (("upstream", original), ("modular", converted)):
        torchaudio.save(args.output / f"{name}.wav", audio.reshape(1, -1).cpu(), 16000)
    result = json.dumps(report, indent=2)
    (args.output / "report.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
