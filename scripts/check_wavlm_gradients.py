"""One-step pretrained WavLM autograd check; no training artifacts are saved."""

import argparse
import hashlib

import torch
from torch import nn

from quick_convert.components.ssl.wavlm import WavLMContentEncoder
from quick_convert.data import AudioBatch, AudioSample
from quick_convert.systems.reconstruction import SSLReconstructionSystem


def weight_digest(module):
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.digest()


def check_step(system, batch, *, trainable):
    encoder = system.online_encoders["content"]
    system.configure_online_encoders({"content": encoder}, ("content",) if trainable else ())
    # Eval mode keeps this plumbing check deterministic, without disabling autograd.
    system.eval()
    system.zero_grad(set_to_none=True)
    before = weight_digest(encoder)
    resource = system.get_resource(batch, "content")
    features = resource.values
    if features.requires_grad != trainable:
        raise RuntimeError("Unexpected encoder feature requires_grad flag.")
    # Frozen forwards use inference_mode; clone outside it for the trainable head.
    if not trainable:
        features = features.clone()
    features = features[:, : int(resource.lengths[0])]
    head = nn.Linear(features.shape[-1], 1).to(features.device)
    head_before = head.weight.detach().clone()
    optimizer = torch.optim.SGD([*system.parameters(), *head.parameters()], lr=1e-3)
    prediction = head(features).squeeze(-1)
    loss = (prediction - 1.0).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite loss.")
    loss.backward()
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Nonfinite encoder gradient.")
    nonzero = sum(bool(g.count_nonzero()) for g in gradients)
    if trainable and not nonzero:
        raise RuntimeError("No nonzero encoder gradients.")
    if not trainable and gradients:
        raise RuntimeError("Frozen encoder received gradients.")
    optimizer.step()
    changed = weight_digest(encoder) != before
    if changed != trainable:
        raise RuntimeError(f"Unexpected encoder update: changed={changed}.")
    if torch.equal(head_before, head.weight.detach()):
        raise RuntimeError("Regression head did not update.")
    mode = "trainable" if trainable else "frozen"
    print(f"PASS {mode}: loss={loss.item():.6f}, nonzero_grad_tensors={nonzero}, encoder_changed={changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Short audio clip (roughly 1–3 seconds).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-name", default="microsoft/wavlm-large")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(115)
    encoder = WavLMContentEncoder(
        model_name=args.model_name,
        layer=-1,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    sample = AudioSample.from_path(args.audio).load_audio(target_sr=encoder.sample_rate, mono=True, device="cpu")
    batch = AudioBatch.from_samples([sample])
    # Only get_resource is exercised; no reconstruction decoder is needed.
    system = SSLReconstructionSystem(decoder=nn.Identity(), online_encoders={"content": encoder})
    check_step(system, batch, trainable=False)
    check_step(system, batch, trainable=True)


if __name__ == "__main__":
    main()
