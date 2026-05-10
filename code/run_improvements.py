"""
Re-run FineXL on cached models with the improved decomposition variants.

For each (model in manifest):
  1. Load 25 paired (base, personalized) images and compute V_div from CLIP
     image features.
  2. Use a fixed style-vocabulary as the candidate concept pool (50 adjectives).
     This is a fair, controlled test isolating the decomposition step from the
     VLM proposer (which we held identical between FineXL and baselines).
  3. Run four decomposition variants:
       - LSQ (paper's Algorithm 1, no centering)
       - LSQ + centering (T1)
       - NNLS (T4, non-negative sparse)
       - NNLS + centering (T1+T4)
  4. Save the resulting explanation vector for each variant.

Then run the model-selection task with each variant and compare LPIPS / accuracy.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
from glob import glob
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from finexl import CLIPEncoder, divergence_vector_from_features, concept_to_vector
from finexl_full import PROBE_PROMPTS
from finexl_improved import improved_decompose


# A small, deliberately diverse style vocabulary covering the 4 ground-truth
# styles and many distractors. Fixed across all models.
STYLE_VOCAB = [
    # ground-truth-related
    "ukiyo-e", "woodblock print", "japanese print", "flat color",
    "watercolor", "soft watercolor", "pastel", "wash painting",
    "pixel art", "8-bit", "16-bit", "retro game",
    "oil painting", "impasto", "rembrandt", "renaissance",
    # distractors / generic
    "vibrant", "muted", "monochrome", "high contrast",
    "minimalist", "abstract", "geometric", "cartoonish",
    "photorealistic", "blurry", "sharp", "detailed",
    "vintage", "modern", "futuristic", "cyberpunk",
    "dreamy", "surreal", "dark", "bright",
    "sketch", "ink drawing", "charcoal", "graphite",
    "stylized", "illustrative", "graphic novel", "anime",
    "psychedelic", "art nouveau", "art deco", "pop art",
    "soft", "hard edges",
]


def load_manifest():
    with open("/home/haoming/finexl/runs/eval/manifest.json") as f:
        return json.load(f)


def main():
    manifest = load_manifest()
    print(f"loading CLIP...", flush=True)
    enc = CLIPEncoder("ViT-B-32", "openai")

    # Pre-compute concept vectors once (they don't depend on the model)
    print(f"computing f(C) for {len(STYLE_VOCAB)} concepts...", flush=True)
    concept_vecs = {c: concept_to_vector(enc, c, PROBE_PROMPTS) for c in STYLE_VOCAB}

    # Compute the geometry stat: pairwise cos and dot with mean
    F = np.stack([concept_vecs[c] for c in STYLE_VOCAB], axis=0)
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    pair_cos = F_norm @ F_norm.T
    np.fill_diagonal(pair_cos, np.nan)
    print(f"  concept-pair cos: mean={np.nanmean(pair_cos):.3f} "
          f"min={np.nanmin(pair_cos):+.3f} max={np.nanmax(pair_cos):+.3f}")
    mu = F.mean(axis=0)
    print(f"  ‖μ‖ / mean ‖f(C)‖ = "
          f"{np.linalg.norm(mu)/F.shape[0]/np.linalg.norm(F,axis=1).mean():.3f}")

    out = {}
    for i, m in enumerate(manifest):
        base_paths = sorted(glob(os.path.join(
            "/home/haoming/finexl/runs/eval/base", "*.png")))[:25]
        pers_paths = sorted(glob(os.path.join(m["eval_dir"], "*.png")))[:25]
        if len(base_paths) < 5 or len(pers_paths) < 5:
            continue
        base_imgs = [Image.open(p).convert("RGB") for p in base_paths]
        pers_imgs = [Image.open(p).convert("RGB") for p in pers_paths]
        base_f = enc.encode_images(base_imgs)
        pers_f = enc.encode_images(pers_imgs)
        Vdiv = divergence_vector_from_features(base_f, pers_f)

        cv = [concept_vecs[c] for c in STYLE_VOCAB]
        rec = {"V_div_norm": float(np.linalg.norm(Vdiv))}
        for use_centering, use_nnls, name in [
            (False, False, "lsq_orig"),
            (True,  False, "lsq_centered"),
            (False, True,  "nnls_orig"),
            (True,  True,  "nnls_centered"),
        ]:
            res = improved_decompose(
                Vdiv, STYLE_VOCAB, cv,
                use_centering=use_centering, use_nnls=use_nnls,
                keep_top_k=8,
            )
            rec[name] = {
                "concepts": res.concepts,
                "weights": res.weights,
                "decomp_error": res.decomp_error,
                "vec": res.explanation_vec.tolist(),
            }
        out[m["mid"]] = rec
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(manifest)}] {m['mid']}  "
                  f"|Vdiv|={rec['V_div_norm']:.3f}", flush=True)

    out_path = "/home/haoming/finexl/runs/results/improved.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
