VENV="$(dirname "$(dirname "$(uv python find)")")"

sed -i '' \
's/from hydra\.experimental import compose, initialize/from hydra import compose, initialize/g' \
"$VENV/lib/python3.9/site-packages/fairseq/dataclass/utils.py"