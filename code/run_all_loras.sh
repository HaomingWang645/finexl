#!/bin/bash
# Train LoRA for the remaining 3 styles, sequentially on GPU 3.
set -e
cd /home/haoming/finexl

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/haoming/finexl/models/hf_cache

for STYLE in watercolor pixel_art oil_painting; do
    echo "=== training $STYLE ==="
    /tmp/finexl_venv/bin/python code/train_lora.py \
        --style_dir data/styles/$STYLE \
        --out runs/lora/$STYLE \
        --snapshots 25 50 75 100 150 200 300 400 \
        --caption "art" \
        2>&1 | tee runs/lora_$STYLE.log
done
echo "=== all single-aspect LoRAs done ==="
