FILE="$(dirname "$(dirname "$(uv python find)")")/lib/python3.9/site-packages/fairseq/dataclass/utils.py"

if [[ "$OSTYPE" == darwin* ]]; then
  sed -i '' 's/from hydra\.experimental import compose, initialize/from hydra import compose, initialize/g' "$FILE"
else
  sed -i 's/from hydra\.experimental import compose, initialize/from hydra import compose, initialize/g' "$FILE"
fi