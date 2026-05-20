# quick_convert/cli/test.py

import hydra
import torch

from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="run/test_chatterbox_spectrogram_generator",
)
def run(cfg):

    model = instantiate(cfg.component).to(cfg.device)

    content = torch.randn(
        cfg.batch_size,
        cfg.content_length,
        cfg.content_dim,
        device=cfg.device,
    )

    content_lengths = torch.full(
        (cfg.batch_size,),
        cfg.content_length,
        dtype=torch.long,
        device=cfg.device,
    )

    speaker_embeddings = torch.randn(
        cfg.batch_size,
        cfg.speaker_dim,
        device=cfg.device,
    )

    with torch.inference_mode():
        mel = model(
            content_features=content,
            content_lengths=content_lengths,
            speaker_embedding=speaker_embeddings,
            n_timesteps=cfg.n_timesteps,
        )

    print("SUCCESS")
    print("mel shape:", mel.shape)


if __name__ == "__main__":
    run()
