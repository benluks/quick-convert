from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger


@dataclass
class ReconstructedAudio:
    original_audio: torch.Tensor | None = None
    reconstructed_audio: torch.Tensor | None = None
    original_mel: torch.Tensor | None = None
    reconstructed_mel: torch.Tensor | None = None

    audio_lengths: torch.Tensor | None = None
    mel_lengths: torch.Tensor | None = None

    original_sample_rate: int | None = None
    reconstructed_sample_rate: int | None = None
    ids: list[str] | None = None


class MediaLogger(ABC):
    @abstractmethod
    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        step: int,
    ) -> None: ...

    @abstractmethod
    def log_images(
        self,
        images: dict[str, torch.Tensor],
        *,
        step: int,
    ) -> None: ...

    @abstractmethod
    def log_bar(
        self,
        key: str,
        values: torch.Tensor,
        *,
        labels: list[str] | None = None,
        step: int,
    ) -> None: ...

    @abstractmethod
    def log_reconstructed_audio(
        self,
        key: str,
        media: ReconstructedAudio,
        *,
        step: int,
        max_samples: int = 8,
    ) -> None: ...


class WandbMediaLogger(MediaLogger):
    import wandb

    def __init__(self, logger: WandbLogger):

        try:
            import wandb
        except ImportError as exc:
            raise ImportError("WandbMediaLogger requires wandb. Install it with `pip install wandb`.") from exc

        self.logger = logger
        self.wandb = wandb

    def log_audio(
        self,
        key,
        audio,
        *,
        sample_rate,
        step,
    ):
        audio = audio.detach().cpu().float().reshape(-1)

        self.logger.experiment.log(
            {
                key: self.wandb.Audio(
                    audio.numpy(),
                    sample_rate=sample_rate,
                )
            },
            step=step,
        )

    def log_images(
        self,
        images,
        *,
        step,
    ):
        self.logger.experiment.log(
            {key: self.wandb.Image(image.detach().cpu().float().numpy()) for key, image in images.items()},
            step=step,
        )

    def log_bar(
        self,
        key,
        values,
        *,
        labels=None,
        step,
    ):
        values = values.detach().cpu().float().reshape(-1)

        if labels is None:
            labels = [str(i) for i in range(len(values))]

        table = self.wandb.Table(
            data=[[label, value.item()] for label, value in zip(labels, values)],
            columns=["layer", "weight"],
        )

        self.logger.experiment.log(
            {
                key: self.wandb.plot.bar(
                    table,
                    "layer",
                    "weight",
                    title=key,
                )
            },
            step=step,
        )

    def _prepare_mel(
        self,
        mel: torch.Tensor,
        *,
        vmin: float = -12.0,
        vmax: float = 2.0,
    ) -> torch.Tensor:

        if mel is None:
            return None
        # images have to be in the [0, 255] range in wandb
        mel = mel.detach().cpu().float()

        mel = (mel - vmin) / (vmax - vmin)
        mel = mel.clamp(0, 1)

        return (mel * 255).to(torch.uint8)

    def log_reconstructed_audio(
        self,
        key,
        media,
        *,
        step,
        max_samples=8,
    ):
        table = self.wandb.Table(
            columns=[
                "id",
                "original_audio",
                "reconstructed_audio",
                "original_mel",
                "reconstructed_mel",
            ]
        )

        n = min(
            max_samples,
            len(media.reconstructed_audio),
        )

        prepared_original_mels = self._prepare_mel(media.original_mel)
        prepared_reconstructed_mels = self._prepare_mel(media.reconstructed_mel)
        for i in range(n):
            utt_id = media.ids[i] if media.ids is not None else str(i)

            original_audio = None
            if media.original_audio is not None:
                original_audio = self.wandb.Audio(
                    media.original_audio[i].detach().cpu().float().reshape(-1).numpy()[: media.audio_lengths[i]],
                    sample_rate=media.original_sample_rate,
                )

            reconstructed_audio = None
            if media.reconstructed_audio is not None:
                reconstructed_audio = self.wandb.Audio(
                    media.reconstructed_audio[i].detach().cpu().float().reshape(-1).numpy()[: media.audio_lengths[i]],
                    sample_rate=media.reconstructed_sample_rate,
                )

            if media.original_mel is not None:
                original_mel = self.wandb.Image(
                    prepared_original_mels[i].detach().cpu().float().numpy()[..., : media.mel_lengths[i]]
                )

            if media.reconstructed_mel is not None:
                reconstructed_mel = self.wandb.Image(
                    prepared_reconstructed_mels[i].detach().cpu().float().numpy()[..., : media.mel_lengths[i]]
                )

            table.add_data(
                utt_id,
                original_audio,
                reconstructed_audio,
                original_mel,
                reconstructed_mel,
            )

        self.logger.experiment.log(
            {key: table},
            step=step,
        )


class TensorBoardMediaLogger(MediaLogger):
    def __init__(self, logger: TensorBoardLogger):
        self.logger = logger

    def log_audio(
        self,
        key,
        audio,
        *,
        sample_rate,
        step,
    ):
        audio = audio.detach().cpu().float().reshape(1, -1)

        self.logger.experiment.add_audio(
            key,
            audio,
            global_step=step,
            sample_rate=sample_rate,
        )

    def log_images(
        self,
        images,
        *,
        step,
    ):
        for key, image in images.items():
            image = image.detach().cpu().float()

            if image.ndim == 2:
                image = image.unsqueeze(0)

            self.logger.experiment.add_image(
                key,
                image,
                global_step=step,
            )

    def log_bar(
        self,
        key,
        values,
        *,
        labels=None,
        step,
    ):
        import matplotlib.pyplot as plt

        values = values.detach().cpu().float().reshape(-1)

        if labels is None:
            labels = [str(i) for i in range(len(values))]

        fig, ax = plt.subplots()

        ax.bar(labels, values.numpy())
        ax.set_xlabel("Layer")
        ax.set_ylabel("Weight")
        ax.set_title(key)
        ax.set_ylim(0, 1)

        self.logger.experiment.add_figure(
            key,
            fig,
            global_step=step,
        )

        plt.close(fig)


class NullMediaLogger(MediaLogger):
    def log_audio(
        self,
        key: str,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        step: int,
    ) -> None:
        pass

    def log_images(
        self,
        images: dict[str, torch.Tensor],
        *,
        step: int,
    ) -> None:
        pass


def make_media_logger(logger: Logger) -> MediaLogger:
    if isinstance(logger, WandbLogger):
        return WandbMediaLogger(logger)

    if isinstance(logger, TensorBoardLogger):
        return TensorBoardMediaLogger(logger)

    raise TypeError(f"Media logging is not supported for {type(logger).__name__}")
