from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

OmegaConf.register_new_resolver("add", lambda x, y: int(x) + int(y))
OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
OmegaConf.register_new_resolver("bool", lambda x: bool(x))

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name=None,
        overrides=[
            "+anonymizer=controllable_rvq",
            "+architecture=controllable_rvq",
            "+components/ssl@anonymizer.content_encoder=w2vbert",
            "device=mps",
        ],
    )

model = instantiate(cfg.anonymizer)

ckpt = torch.load("last.ckpt", map_location="mps", weights_only=False)
state = ckpt["state_dict"]

missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
