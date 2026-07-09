from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.data.resources.base import ResourceCollection
from quick_convert.data.resources.providers import PathResourceProvider
from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.encoder_decoder.controllable_rvq import ControllableRVQTrainingModule

OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
OmegaConf.register_new_resolver("bool", lambda x: bool(x))

device = "mps"
with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name=None,
        overrides=[
            "+anonymizer=controllable_rvq",
            "+architecture=controllable_rvq",
            "+components/ssl@anonymizer.online_encoders.content=w2vbert",
            f"device={device}",
        ],
    )

model: ControllableRVQTrainingModule = instantiate(cfg.anonymizer)
missing, unexpected = model.load_for_inference(checkpoint_path="last.ckpt", map_location=device, strict=False)

model = model.to(device)
model.eval()

resources_path = "outputs/precomputed/librispeech"
emo_provider = PathResourceProvider(
    kind="torch_tensor", name="emo2vec", path_template=resources_path + "/emo2vec/test-other/{sample.utt_id}.pt"
)
batch = AudioBatch.from_paths(
    [
        "/Users/ben/LibriSpeech/test-other/1688/142285/1688-142285-0000.flac",
        "/Users/ben/LibriSpeech/test-other/1688/142285/1688-142285-0001.flac",
        "/Users/ben/LibriSpeech/test-other/1688/142285/1688-142285-0002.flac",
        "/Users/ben/LibriSpeech/test-other/1688/142285/1688-142285-0003.flac",
    ],
    resource_providers=[emo_provider],
    device=device,
)

wavs = model.inference(batch)
