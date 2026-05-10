#!/bin/bash
# Train multi-aspect LoRAs by mixing style training data, for Table 3.
# - 2-aspect: 6 pairs of styles, 50/50 mix
# - 3-aspect: 4 triples of styles, 33/33/33 mix
# - 4-aspect: 1 quad, 25/25/25/25 mix
# Each at 200 training steps (paper's typical "intermediate" level).
set -e
cd /home/haoming/finexl

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/home/haoming/finexl/models/hf_cache

train_one () {
    local out=$1; shift
    local dirs=$1; shift
    if [ -d "runs/lora_multi/$out/step_0200" ]; then
        echo "SKIP $out (exists)"; return
    fi
    echo "=== $out  mix=$dirs ==="
    /tmp/finexl_venv/bin/python code/train_lora.py \
        --mixed_dirs "$dirs" \
        --style_dir /dev/null \
        --out runs/lora_multi/$out \
        --snapshots 200 \
        --caption "art" \
        2>&1 | tee runs/lora_multi_$out.log | tail -3
}

# 2-aspect pairs
for PAIR in "ukiyo+watercolor" "ukiyo+pixel_art" "ukiyo+oil_painting" \
            "watercolor+pixel_art" "watercolor+oil_painting" \
            "pixel_art+oil_painting"; do
    A=${PAIR%+*}; B=${PAIR#*+}
    train_one "${A}_${B}" "data/styles/$A,data/styles/$B"
done

# 3-aspect triples
for TRI in "ukiyo+watercolor+pixel_art" "ukiyo+watercolor+oil_painting" \
           "ukiyo+pixel_art+oil_painting" "watercolor+pixel_art+oil_painting"; do
    a=${TRI%%+*}; rest=${TRI#*+}; b=${rest%%+*}; c=${rest#*+}
    train_one "${a}_${b}_${c}" "data/styles/$a,data/styles/$b,data/styles/$c"
done

# 4-aspect single
train_one "ukiyo_watercolor_pixel_art_oil_painting" \
    "data/styles/ukiyo,data/styles/watercolor,data/styles/pixel_art,data/styles/oil_painting"

echo "=== all multi-aspect LoRAs done ==="
