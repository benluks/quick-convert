from .base_dataset import BaseDataset


class ClacDataset(BaseDataset):
    def __init__(
        self,
        root=None,
        splits=None,
        file_format=None,
        paths=None,
        load=False,
        sample_rate=48000,
    ):
        super().__init__(
            root=root,
            splits=splits,
            file_format=file_format,
            paths=paths,
            load=load,
        )
