"""
Iteration 4: image-domain concept directions (T3) + sweep around F_18_nnls_eval.

For each top concept, we replace text-derived f(C) with the *image*-domain
direction:
    f_img(C) = mean_t Enc_img(SD_base("{t}, {C} style")) - mean_t Enc_img(SD_base("{t}"))
This eliminates the CLIP text-vs-image alignment gap that the diagnostic
identified as the root cause.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
from glob import glob
from PIL import Image
from diffusers import StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(__file__))
from finexl import CLIPEncoder, divergence_vector_from_features, concept_to_vector
from finexl_full import PROBE_PROMPTS
from iterate3 import EVAL_PROMPTS
from iterate_improvements import (
    STYLE_VOCAB_150, load_manifest, load_explanations,
    cosine, run_selection, make_lpips_fn, variant_decompose,
    build_pca_whitener,
)


# Top concepts (by relevance / coverage of the 4 ground-truth styles)
# we'll compute image-domain f_img(C) for these
TOP_CONCEPTS = [
    "ukiyo-e", "woodblock print", "japanese print", "flat color",
    "watercolor", "soft watercolor", "pastel", "wash painting",
    "pixel art", "8-bit", "16-bit", "retro game",
    "oil painting", "impasto", "rembrandt", "renaissance",
    "vibrant", "muted", "monochrome", "high contrast",
    "minimalist", "abstract", "cartoonish", "stylized",
    "photorealistic", "blurry", "sharp", "detailed",
    "vintage", "anime",
]


def compute_image_domain_directions(enc, sd_pipe, concepts, prompts, n_per=2):
    """For each concept C, compute mean image-feature shift over prompts."""
    print(f"  generating base images for {len(prompts)} prompts...", flush=True)
    base_imgs = {}
    for i, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(9000 + i)
        img = sd_pipe(p, num_inference_steps=20, height=512, width=512,
                      generator=g).images[0]
        base_imgs[p] = img
    base_f = enc.encode_images(list(base_imgs.values())).mean(axis=0)

    out = {}
    for j, c in enumerate(concepts):
        imgs = []
        for i, p in enumerate(prompts):
            g = torch.Generator("cuda").manual_seed(9000 + i + 137 * j)
            img = sd_pipe(f"{p}, {c} style", num_inference_steps=20,
                          height=512, width=512, generator=g).images[0]
            imgs.append(img)
        cf = enc.encode_images(imgs).mean(axis=0)
        out[c] = cf - base_f
        if (j + 1) % 5 == 0:
            print(f"  [{j+1}/{len(concepts)}] f_img({c})", flush=True)
    return out


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

    # Compute image-domain directions for TOP_CONCEPTS
    cache_path = "/home/haoming/finexl/runs/results/f_img_cache.json"
    if os.path.exists(cache_path):
        print(f"loading cached f_img from {cache_path}", flush=True)
        with open(cache_path) as f:
            f_img_dict = {k: np.array(v) for k, v in json.load(f).items()}
    else:
        print("loading SD 1.5 for image-domain f(C)...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir="/home/haoming/finexl/models/hf_cache",
            torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        # Use the first 4 EVAL prompts (domain-matched)
        f_img_dict = compute_image_domain_directions(
            enc, pipe, TOP_CONCEPTS, EVAL_PROMPTS[:4], n_per=1)
        with open(cache_path, "w") as f:
            json.dump({k: v.tolist() for k, v in f_img_dict.items()}, f)
        print(f"saved f_img to {cache_path}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    # Text-domain f(C) for full vocab
    print(f"f_text(C) for {len(STYLE_VOCAB_150)} concepts (eval probes)...", flush=True)
    fC_text = {c: concept_to_vector(enc, c, EVAL_PROMPTS) for c in STYLE_VOCAB_150}

    # Hybrid f(C): image-domain for TOP_CONCEPTS, text-domain for others
    fC_hybrid = dict(fC_text)
    for c in TOP_CONCEPTS:
        if c in f_img_dict:
            fC_hybrid[c] = f_img_dict[c]

    # Pure image-domain (TOP_CONCEPTS only)
    fC_img_only = {c: f_img_dict[c] for c in TOP_CONCEPTS if c in f_img_dict}

    # F matrices
    vocab_full = STYLE_VOCAB_150
    F_text = np.stack([fC_text[c] for c in vocab_full], axis=0)
    F_hybrid = np.stack([fC_hybrid[c] for c in vocab_full], axis=0)
    vocab_img = list(fC_img_only.keys())
    F_img = np.stack([fC_img_only[c] for c in vocab_img], axis=0)

    # Whiteners
    PC_COUNTS = [12, 15, 18, 22, 30]
    print("building whiteners ...", flush=True)
    W_text = {n: build_pca_whitener(F_text, n_pcs=n, eps=1e-3) for n in PC_COUNTS}
    W_hybrid = {n: build_pca_whitener(F_hybrid, n_pcs=n, eps=1e-3) for n in PC_COUNTS}
    W_img = {n: build_pca_whitener(F_img, n_pcs=min(n, F_img.shape[0]-1), eps=1e-3)
             for n in PC_COUNTS}

    # Variants
    variants = {}
    for n in PC_COUNTS:
        variants[f"text_{n}_nnls"] = ("text", n, "nnls", False)
        variants[f"hybrid_{n}_nnls"] = ("hybrid", n, "nnls", False)
        variants[f"img_{n}_nnls"]  = ("img",  n, "nnls", False)
        variants[f"hybrid_{n}_lsq"]  = ("hybrid", n, "lsq",  False)
    # Add centering combos around best
    for n in [12, 15, 18]:
        variants[f"hybrid_{n}_nnls_c"] = ("hybrid", n, "nnls", True)

    print(f"running {len(variants)} variants × {len(manifest)} models...", flush=True)
    vecs = {v: {} for v in variants}
    for m in manifest:
        mid = m["mid"]
        Vdiv = Vdiv_per_model[mid]
        for v_name, (kind, n, dec, center) in variants.items():
            if kind == "text":
                cv = list(F_text); W = W_text[n]
            elif kind == "hybrid":
                cv = list(F_hybrid); W = W_hybrid[n]
            else:
                cv = list(F_img); W = W_img[n]
            _, _, _, ev = variant_decompose(
                Vdiv, cv, use_centering=center,
                use_nnls=(dec == "nnls"), whitening=W)
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
        for v, r in items[:6]:
            print(f"  {v:22s}  ss={r['same_style']:.2f}  sm={r['same_mix']:.2f}  lpips={r['lpips']:.3f}")

    with open("/home/haoming/finexl/runs/results/iterate4.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
