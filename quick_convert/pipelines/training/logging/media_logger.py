import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger


@dataclass
class ReconstructedAudio:
    original_audio: torch.Tensor | None = None
    reconstructed_audio: torch.Tensor | None = None
    original_mel: torch.Tensor | None = None
    reconstructed_mel: torch.Tensor | None = None

    original_audio_lengths: torch.Tensor | None = None
    reconstructed_audio_lengths: torch.Tensor | None = None
    original_mel_lengths: torch.Tensor | None = None
    reconstructed_mel_lengths: torch.Tensor | None = None

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
        key,
        values,
        item_name: str,
        value_name: str,
        *,
        item_labels=None,
        step,
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

    @abstractmethod
    def log_text(
        self,
        texts: dict[str, str],
        *,
        step: int,
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

    def _log(self, data, *, step):
        try:
            self.logger.experiment.log(data, step=step)
        except TimeoutError as exc:
            warnings.warn(f"W&B logging timed out at step {step}: {exc}", stacklevel=2)

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
        item_name: str,
        value_name: str,
        *,
        item_labels=None,
        step,
    ):
        values = values.detach().cpu().float().reshape(-1)

        if item_labels is None:
            item_labels = list(range(len(values)))

        table = self.wandb.Table(
            data=[[label, value.item()] for label, value in zip(item_labels, values, strict=True)],
            columns=[item_name, value_name],
        )

        bar = self.wandb.plot.bar(
            table,
            item_name,
            value_name,
            title=key,
        )

        self._log({key: bar}, step=step)

    def log_heatmap(
        self,
        key,
        values,
        *,
        step,
        x_labels=None,
        y_labels=None,
        annotate=False,
        vmin=None,
        vmax=None,
    ):

        values = values.detach().cpu().float()

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            values.numpy(),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        if x_labels is not None:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels)

        if y_labels is not None:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)

        if annotate:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                    )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        self.log_figure(key, fig, step=step)
        plt.close(fig)

    def log_figure(self, key, figure, *, step):
        self.logger.experiment.log(
            {key: self.wandb.Image(figure)},
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
        media: ReconstructedAudio,
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
            len(media.original_audio),
        )

        prepared_original_mels = self._prepare_mel(media.original_mel)
        prepared_reconstructed_mels = self._prepare_mel(media.reconstructed_mel)
        for i in range(n):
            utt_id = media.ids[i] if media.ids is not None else str(i)

            original_audio = None
            if media.original_audio is not None:
                original_audio = self.wandb.Audio(
                    media.original_audio[i]
                    .detach()
                    .cpu()
                    .float()
                    .reshape(-1)
                    .numpy()[: media.original_audio_lengths[i]],
                    sample_rate=media.original_sample_rate,
                )

            reconstructed_audio = None
            if media.reconstructed_audio is not None:
                reconstructed_audio = self.wandb.Audio(
                    media.reconstructed_audio[i]
                    .detach()
                    .cpu()
                    .float()
                    .reshape(-1)
                    .numpy()[: media.reconstructed_audio_lengths[i]],
                    sample_rate=media.reconstructed_sample_rate,
                )

            original_mel = None
            reconstructed_mel = None

            if media.original_mel is not None:
                original_mel = self.wandb.Image(
                    prepared_original_mels[i].detach().cpu().float().numpy()[..., : media.original_mel_lengths[i]]
                )

            if media.reconstructed_mel is not None:
                reconstructed_mel = self.wandb.Image(
                    prepared_reconstructed_mels[i]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()[..., : media.reconstructed_mel_lengths[i]]
                )

            table.add_data(
                utt_id,
                original_audio,
                reconstructed_audio,
                original_mel,
                reconstructed_mel,
            )

        self._log(
            {key: table},
            step=step,
        )

    def log_text(
        self,
        texts: dict[str, str],
        *,
        step: int,
    ) -> None:
        self._log(
            texts,
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
