import sys
from pathlib import Path

from hyperpyyaml import load_hyperpyyaml
from quick_convert.pipelines.anonymization.pipeline import Pipeline


def load_config():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m scripts.anonymize config.yaml [key=value ...]"
        )

    config_path = Path(sys.argv[1])
    overrides = sys.argv[2:]

    with open(config_path) as f:
        hparams = load_hyperpyyaml(f, overrides)

    return hparams


def main():
    hparams = load_config()

    pipeline = Pipeline(hparams["anonymizer"])
    pipeline.anonymize_dir(**hparams["run"])


if __name__ == "__main__":
    main()
