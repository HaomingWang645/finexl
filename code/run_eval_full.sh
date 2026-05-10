#!/bin/bash
# Run the full Table 1 evaluation: explanations for all models, then
# selection task across multiple "level subset" configurations.
set -e
cd /home/haoming/finexl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/haoming/finexl/models/hf_cache

PY=/tmp/finexl_venv/bin/python

# 1. Build manifest
$PY code/build_manifest.py

# 2. Compute explanations for every model (cached as explanations.json)
mkdir -p runs/results
$PY code/eval_pipeline.py \
    --manifest runs/eval/manifest.json \
    --base_eval_dir runs/eval/base \
    --out_dir runs/results/full \
    --explanations_only

# 3. Run selection task for each level subset (Table 1 columns)
for SUBSET in "25 100 400" "25 75 100 200 400" "25 50 75 100 150 200 300 400"; do
    NAME=$(echo "$SUBSET" | tr ' ' '_')
    OUTD="runs/results/sel_${NAME}"
    mkdir -p "$OUTD"
    cp runs/results/full/explanations.json "$OUTD/"
    $PY code/eval_pipeline.py \
        --manifest runs/eval/manifest.json \
        --base_eval_dir runs/eval/base \
        --out_dir "$OUTD" \
        --levels_subset $SUBSET
done

echo "=== full eval done ==="
