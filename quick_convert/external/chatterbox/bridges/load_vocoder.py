from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..s3gen.s3gen import S3Token2Wav

REPO_ID = "ResembleAI/chatterbox"


def load_vocoder(device=None):
    safetensors_path = hf_hub_download(repo_id=REPO_ID, filename="s3gen.safetensors")
    vocoder = S3Token2Wav().mel2wav
    state_dict = load_file(safetensors_path)

    # vocoder_state = {k: v for k, v in state_dict.items() if not k.startswith("tokenizer")}

    generator_keys = {k.split(".", maxsplit=1)[1]: v for k, v in state_dict.items() if k.startswith("mel2wav")}

    incompatible_keys = vocoder.load_state_dict(generator_keys, strict=True)
    vocoder.eval()
    return vocoder, incompatible_keys
