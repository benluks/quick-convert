import torch

from .masking import mask_pad


def configure_device(device: str | torch.device | None = None) -> torch.device:
    """
    Configure the device for PyTorch.

    Args:
        device:
            The device to use. If None, will use CUDA if available, else CPU.

    Returns:
        The configured device.
    """
    if device is None:
        return torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(device)


__all__ = ["configure_device", "mask_pad"]
