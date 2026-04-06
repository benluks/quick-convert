from pathlib import Path

from .base_dataset import BaseDataset


class ClacDataset(BaseDataset):
    def __init__(
        self,
        root=None,
        splits=None,
        file_format=None,
        paths=None,
        load=False,
        return_spkid=True,
    ):
        super().__init__(
            root=root,
            splits=splits,
            file_format=file_format,
            paths=paths,
            load=load,
            return_spkid=return_spkid,
        )

    def get_spkid(self, wav_path: Path):
        # default spk_id logic: immediate parent dir
        return wav_path.stem
