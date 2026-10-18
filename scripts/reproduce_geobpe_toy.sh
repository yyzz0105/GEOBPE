#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/users/wwygrseth/.conda_envs/GeoBPE-repro/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/reproduce_geobpe_toy}"
ENCODE_DIR="$RUN_DIR/encode"
INDUCE_DIR="$RUN_DIR/induce"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export SLURM_CPUS_PER_TASK=0
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

mkdir -p "$RUN_DIR"

"$PYTHON_BIN" bin/encode.py \
  --data-dir train --toy 2 --pad 512 \
  --save-dir "$ENCODE_DIR" \
  --log-dir "$ENCODE_DIR/logs" \
  --bins '1-500' --bin-strategy histogram \
  --res-init true --free-bonds true --rmsd-super-res true \
  --num-p '2-8:3-16' --max-num-strucs 2 --p-min-size 0 \
  --glue-opt true --glue-opt-prior 1.0 --glue-opt-every 1 \
  --glue-opt-method all --max-iter 1 --save-every 1 \
  --plot-every 999 --num-ref 2 --num-vis 0 --seed 0

"$PYTHON_BIN" scripts/verify_geobpe_roundtrip.py \
  --bpe-pkl "$ENCODE_DIR/bpe_post_init.pkl" \
  --output-dir "$ENCODE_DIR/roundtrip" \
  --chains 0 1

"$PYTHON_BIN" bin/induce.py \
  --src-pkl "$ENCODE_DIR/bpe_post_init.pkl" \
  --base-dir "$ROOT" --save-dir "$INDUCE_DIR" \
  --log-dir "$INDUCE_DIR/logs" \
  --data-dir validation --toy 2 --pad 512 \
  --processed false --debug --append

"$PYTHON_BIN" bin/train.py \
  --checkpoint_path "$INDUCE_DIR/bpe_post_init.pkl" \
  --task pretrain --device cpu --skip_sctm \
  --seed 0 --d_model 64 --num_layers 1 --num_heads 2 --d_ff 256 \
  --batch_size 2 --lr 1e-3 --epochs 1 --max_len 257 \
  --num_samples 2 --temperature 1.0

test -f "$INDUCE_DIR/metrics.json"
test -d "$INDUCE_DIR/sampled_pdb"

"$PYTHON_BIN" scripts/verify_geobpe_roundtrip.py \
  --bpe-pkl "$INDUCE_DIR/bpe_post_init.pkl" \
  --sampled-tokens "$INDUCE_DIR/sampled_tokens.json" \
  --sampled-pdb-dir "$INDUCE_DIR/sampled_pdb" \
  --output-dir "$INDUCE_DIR/roundtrip"

test -f "$INDUCE_DIR/roundtrip/sampled_decode_validation.json"

"$PYTHON_BIN" -m unittest discover -s tests -v

printf 'GeoBPE toy reproduction completed: %s\n' "$RUN_DIR"
