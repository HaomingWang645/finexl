#!/bin/bash
# Train LoRAs for 4 styles with moderate strong config (LR=2e-4, alpha=32).
set -e
cd /home/haoming/finexl

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/haoming/finexl/models/hf_cache

rm -rf runs/lora
for STYLE in ukiyo watercolor pixel_art oil_painting; do
    echo "=== training $STYLE ==="
    /tmp/finexl_venv/bin/python code/train_lora.py \
        --style_dir data/styles/$STYLE \
        --out runs/lora/$STYLE \
        --snapshots 25 50 75 100 150 200 300 400 \
        --lr 2e-4 --rank 16 --alpha 32 \
        --caption "art" \
        2>&1 | tee runs/lora_${STYLE}.log | tail -3
done
echo "=== all single-aspect LoRAs done ==="
