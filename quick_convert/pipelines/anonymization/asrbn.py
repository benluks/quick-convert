import torch

from .base_anonymizer import BaseAnonymizer
from .targets import ASRBNTarget


class ASRBNAnonymizer(BaseAnonymizer[ASRBNTarget]):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "deep-privacy/SA-toolkit",
            "anonymization",
            tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
            trust_repo=True,
        ).to(self.device)
        self.model.eval()

        self.sample_rate = self.sr = 16000

    def set_target(self, target):
        if isinstance(target, int):
            target = str(target)
        self.target = target

    @torch.inference_mode()
    def convert(self, audio_path, target_speaker=None):
        target = target_speaker or self.target

        x = self.load(audio_path)
        return self.model.convert(x.to(self.device), target=target).cpu()


if __name__ == "__main__":
    asrbn = ASRBNAnonymizer()
