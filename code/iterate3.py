"""
Iteration 3: fine-grained PC-count sweep around the F_15_nnls sweet spot,
plus centering on top of PCA whitening, plus a 250-prompt probe set
(domain-matched, larger sample for f(C)).
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
    STYLE_VOCAB_150, load_manifest, load_explanations,
    cosine, run_selection, make_lpips_fn, variant_decompose,
    build_pca_whitener,
)


# Eval prompts (domain-matched probes — same as gen_all_eval.EVAL_PROMPTS)
EVAL_PROMPTS = [
    "a cyclist racing along a coastal road",
    "an elderly woman knitting by a window",
    "a child building a sandcastle on the beach",
    "a busy farmer's market in autumn",
    "a vintage tractor in a wheat field",
    "a cat curled up on a stack of books",
    "two friends laughing at a dinner table",
    "a violinist on a foggy bridge",
    "a hiker watching sunrise from a cliff",
    "a small bakery with fresh bread on display",
    "a fox stepping carefully across snow",
    "a teenager skateboarding under streetlights",
    "a fisherman pulling nets at dusk",
    "an astronaut planting a flag on the moon",
    "a teacher writing equations on a chalkboard",
    "a violinist playing in a quiet park",
    "a chef tossing pasta in a pan",
    "a ballerina spinning on a wooden floor",
    "a deer drinking from a stream",
    "a young girl flying a paper airplane",
    "a fountain in a city plaza at night",
    "a basketball player jumping for a dunk",
    "a samurai standing in a bamboo forest",
    "a couple dancing on a rooftop at sunset",
    "an old typewriter on a cluttered desk",
]


def main():
    manifest = load_manifest()
    print("loading CLIP...", flush=True)
    enc = CLIPEncoder("ViT-B-32", "openai")

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

    # Two probe sets: original (PROBE_PROMPTS) and domain-matched (EVAL_PROMPTS)
    print(f"f(C) for {len(STYLE_VOCAB_150)} concepts (orig probes)...", flush=True)
    fC_orig = {c: concept_to_vector(enc, c, PROBE_PROMPTS) for c in STYLE_VOCAB_150}
    print(f"f(C) for {len(STYLE_VOCAB_150)} concepts (eval-domain probes)...", flush=True)
    fC_eval = {c: concept_to_vector(enc, c, EVAL_PROMPTS) for c in STYLE_VOCAB_150}

    F_orig = np.stack([fC_orig[c] for c in STYLE_VOCAB_150], axis=0)
    F_eval = np.stack([fC_eval[c] for c in STYLE_VOCAB_150], axis=0)

    PC_COUNTS = [5, 8, 10, 12, 15, 18, 22, 30]

    whiteners_orig = {n: build_pca_whitener(F_orig, n_pcs=n, eps=1e-3)
                      for n in PC_COUNTS}
    whiteners_eval = {n: build_pca_whitener(F_eval, n_pcs=n, eps=1e-3)
                      for n in PC_COUNTS}

    # Variants
    variants = {}
    for n in PC_COUNTS:
        variants[f"F_{n}_nnls_orig"] = (n, "nnls", "orig", False)
        variants[f"F_{n}_lsq_orig"]  = (n, "lsq",  "orig", False)
        variants[f"F_{n}_nnls_eval"] = (n, "nnls", "eval", False)
        variants[f"F_{n}_lsq_eval"]  = (n, "lsq",  "eval", False)
        variants[f"F_{n}_nnls_orig_c"] = (n, "nnls", "orig", True)  # +centering

    print(f"running {len(variants)} variants × {len(manifest)} models...", flush=True)
    vecs = {v: {} for v in variants}
    for m in manifest:
        mid = m["mid"]
        Vdiv = Vdiv_per_model[mid]
        for v_name, (n, dec, probe, center) in variants.items():
            cv = list(F_orig if probe == "orig" else F_eval)
            W = whiteners_orig[n] if probe == "orig" else whiteners_eval[n]
            _, _, _, ev = variant_decompose(
                Vdiv, cv,
                use_centering=center,
                use_nnls=(dec == "nnls"),
                whitening=W,
            )
            vecs[v_name][mid] = ev.tolist()

    print("loading LPIPS...", flush=True)
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
        items = sorted(results[label].items(),
                       key=lambda kv: (-kv[1]["same_style"], kv[1]["lpips"]))
        for v, r in items[:8]:
            print(f"  {v:22s}  ss={r['same_style']:.2f}  sm={r['same_mix']:.2f}  lpips={r['lpips']:.3f}")

    with open("/home/haoming/finexl/runs/results/iterate3.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
