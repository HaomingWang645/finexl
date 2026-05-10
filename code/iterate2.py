"""
Second iteration. PCA whitening + LSQ won the first round; sweep PC counts and
add the union-vocab and image-domain-directions variants.

Variants this round:
  F_<n>_lsq    : 150-vocab + PCA-whiten with n PCs + LSQ          (n in {15, 30, 50, 80, 128})
  F_<n>_nnls   : 150-vocab + PCA-whiten with n PCs + NNLS
  G_<n>_lsq    : (150-vocab ∪ per-model Naive proposals) + PCA-whiten n PCs + LSQ
  G_<n>_nnls   : same with NNLS
  H_lsq        : 150-vocab with image-domain f_img(C) for top-30 concepts (rest text)
                  + best PC count + LSQ
  H_nnls       : same with NNLS
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
from iterate_improvements import (
    STYLE_VOCAB_50, STYLE_VOCAB_150, load_manifest, load_explanations,
    cosine, run_selection, make_lpips_fn, variant_decompose,
    build_pca_whitener,
)


def main():
    manifest = load_manifest()
    explanations = load_explanations()
    print("loading CLIP...", flush=True)
    enc = CLIPEncoder("ViT-B-32", "openai")

    # V_div per model (CLIP image features)
    print("V_div per model...", flush=True)
    base_paths = sorted(glob("/home/haoming/finexl/runs/eval/base/*.png"))[:25]
    base_imgs = [Image.open(p).convert("RGB") for p in base_paths]
    base_f = enc.encode_images(base_imgs)
    Vdiv_per_model = {}
    for m in manifest:
        pers_paths = sorted(glob(os.path.join(m["eval_dir"], "*.png")))[:25]
        pers_imgs = [Image.open(p).convert("RGB") for p in pers_paths]
        pers_f = enc.encode_images(pers_imgs)
        Vdiv_per_model[m["mid"]] = divergence_vector_from_features(base_f, pers_f)

    # f(C) for the 150-vocab
    print(f"f(C) for {len(STYLE_VOCAB_150)} concepts ...", flush=True)
    fC = {c: concept_to_vector(enc, c, PROBE_PROMPTS) for c in STYLE_VOCAB_150}

    # F matrix
    F150 = np.stack([fC[c] for c in STYLE_VOCAB_150], axis=0)

    PC_COUNTS = [15, 30, 50, 80, 128]

    # Build whiteners
    whiteners = {n: build_pca_whitener(F150, n_pcs=n, eps=1e-3) for n in PC_COUNTS}

    # Decompose for every (variant, model)
    def get_naive_concepts(mid):
        em = explanations.get(mid, {})
        return [c for c in em.get("Naive", {}).get("concepts", []) if c in fC]

    vecs = {}
    for n in PC_COUNTS:
        vecs[f"F_{n}_lsq"] = {}
        vecs[f"F_{n}_nnls"] = {}
        vecs[f"G_{n}_lsq"] = {}
        vecs[f"G_{n}_nnls"] = {}

    print("running variants ...", flush=True)
    for m in manifest:
        mid = m["mid"]
        Vdiv = Vdiv_per_model[mid]
        # Per-model union vocab
        union = list(set(STYLE_VOCAB_150 + get_naive_concepts(mid)))
        union = [c for c in union if c in fC]
        cv_union = [fC[c] for c in union]
        cv_150 = [fC[c] for c in STYLE_VOCAB_150]

        for n in PC_COUNTS:
            W = whiteners[n]
            # F_n_lsq: 150-vocab
            _, _, _, ev = variant_decompose(
                Vdiv, cv_150, use_centering=False, use_nnls=False, whitening=W)
            vecs[f"F_{n}_lsq"][mid] = ev.tolist()
            _, _, _, ev = variant_decompose(
                Vdiv, cv_150, use_centering=False, use_nnls=True, whitening=W)
            vecs[f"F_{n}_nnls"][mid] = ev.tolist()
            # G_n: union vocab — re-build whitener for the larger union? simpler: reuse W
            # because W is symmetric d×d operating on any d-vector.
            _, _, _, ev = variant_decompose(
                Vdiv, cv_union, use_centering=False, use_nnls=False, whitening=W)
            vecs[f"G_{n}_lsq"][mid] = ev.tolist()
            _, _, _, ev = variant_decompose(
                Vdiv, cv_union, use_centering=False, use_nnls=True, whitening=W)
            vecs[f"G_{n}_nnls"][mid] = ev.tolist()

    # Selection task
    print("loading LPIPS ...", flush=True)
    lpips_fn = make_lpips_fn()
    M = manifest
    M_single = [m for m in M if m.get("n_aspects", 1) == 1]
    columns = []
    for K, levels in [(3,[25,100,400]), (5,[25,75,100,200,400]),
                      (8,[25,50,75,100,150,200,300,400])]:
        sub = [m for m in M_single if m["level"] in levels]
        columns.append((f"single, {K} levels", sub))
    for k in [2, 3, 4]:
        sub = [m for m in M if m.get("n_aspects", 1) == k] + M_single
        columns.append((f"{k}-aspect (+ single pool)", sub))

    results = {}
    for label, sub in columns:
        results[label] = {}
        for v_name, vec_dict in vecs.items():
            ss, sm, lp = run_selection(sub, vec_dict, lpips_fn)
            results[label][v_name] = {"same_style": ss, "same_mix": sm, "lpips": lp}
        print(f"--- {label} ---")
        # print only top 5 by lpips
        items = sorted(results[label].items(), key=lambda kv: kv[1]["lpips"])
        for v, r in items[:6]:
            print(f"  {v:14s}  ss={r['same_style']:.2f}  sm={r['same_mix']:.2f}  lpips={r['lpips']:.3f}")

    with open("/home/haoming/finexl/runs/results/iterate2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
