"""Run the model-selection task using the improved decomposition variants.

For each variant in {lsq_orig, lsq_centered, nnls_orig, nnls_centered},
build the explanation-vector dict over the 43 models and run the same
selection task as before.

Also adds a fifth variant: T2 whitening — applies ZCA whitening to
{f(C)} computed from the same 50-concept vocabulary, then runs LSQ
on the whitened V_div / f(C).
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from finexl import CLIPEncoder, divergence_vector_from_features, concept_to_vector
from finexl_full import PROBE_PROMPTS
from finexl_improved import improved_decompose, decompose_lsq, decompose_nnls
from glob import glob
from PIL import Image


STYLE_VOCAB = [  # same as run_improvements.py
    "ukiyo-e", "woodblock print", "japanese print", "flat color",
    "watercolor", "soft watercolor", "pastel", "wash painting",
    "pixel art", "8-bit", "16-bit", "retro game",
    "oil painting", "impasto", "rembrandt", "renaissance",
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


def whiten_zca(F: np.ndarray, eps: float = 1e-3):
    """ZCA whitening transform W such that cov(W F^T) ≈ I."""
    F = F - F.mean(axis=0, keepdims=True)
    cov = F.T @ F / max(F.shape[0] - 1, 1)         # (d, d) — rank-deficient
    # Add diagonal regularization, then symm-eigh
    cov = cov + eps * np.eye(cov.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return W


def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


def run_selection(models, vec_per_model, lpips_fn):
    same_style_match = 0
    same_mix_match = 0
    lpips_sum = 0.0
    for tgt in models:
        tv = vec_per_model[tgt["mid"]]
        best, best_s = None, -2
        for cand in models:
            if cand["mid"] == tgt["mid"]:
                continue
            s = cosine(tv, vec_per_model[cand["mid"]])
            if s > best_s:
                best, best_s = cand, s
        same_style_match += int(tgt["style"] == best["style"])
        same_mix_match += int(set(tgt["mix"].keys()) == set(best["mix"].keys()))
        lpips_sum += lpips_fn(tgt["eval_dir"], best["eval_dir"])
    n = len(models)
    return (same_style_match / n, same_mix_match / n, lpips_sum / n)


def make_lpips_fn():
    import lpips as lpips_pkg
    import torch
    from torchvision import transforms as T
    net = lpips_pkg.LPIPS(net="alex").to("cuda").eval()
    tx = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    cache = {}
    def load(d):
        if d not in cache:
            paths = sorted(glob(os.path.join(d, "*.png")))[:25]
            cache[d] = [tx(Image.open(p).convert("RGB")).unsqueeze(0).to("cuda")*2-1
                        for p in paths]
        return cache[d]
    @torch.no_grad()
    def score(a, b):
        A, B = load(a), load(b)
        n = min(len(A), len(B))
        return float(np.mean([net(A[i], B[i]).mean().item() for i in range(n)]))
    return score


def main():
    with open("/home/haoming/finexl/runs/eval/manifest.json") as f:
        manifest = json.load(f)
    with open("/home/haoming/finexl/runs/results/improved.json") as f:
        improved = json.load(f)

    # Recompute T2 (whitening) variant since it requires global whitening.
    print("loading CLIP...", flush=True)
    enc = CLIPEncoder("ViT-B-32", "openai")
    print("computing f(C) for whitening...", flush=True)
    F = np.stack([concept_to_vector(enc, c, PROBE_PROMPTS) for c in STYLE_VOCAB], axis=0)
    W = whiten_zca(F, eps=1e-3)
    F_w = (W @ F.T).T  # (k, d) whitened concept vectors

    # Add a "lsq_whitened" entry to each model
    print("computing whitened decomposition per model...", flush=True)
    base_paths = sorted(glob("/home/haoming/finexl/runs/eval/base/*.png"))[:25]
    base_imgs = [Image.open(p).convert("RGB") for p in base_paths]
    base_f = enc.encode_images(base_imgs)
    for i, m in enumerate(manifest):
        if m["mid"] not in improved:
            continue
        pers_paths = sorted(glob(os.path.join(m["eval_dir"], "*.png")))[:25]
        pers_imgs = [Image.open(p).convert("RGB") for p in pers_paths]
        pers_f = enc.encode_images(pers_imgs)
        Vdiv = divergence_vector_from_features(base_f, pers_f)
        Vdiv_w = (W @ Vdiv)
        # LSQ in whitened space
        w, err = decompose_lsq(Vdiv_w, list(F_w))
        order = np.argsort(-np.abs(w))[:8]
        kept = [STYLE_VOCAB[i] for i in order]
        wts = [float(w[i]) for i in order]
        # Build explanation vector in whitened space
        ev = np.zeros_like(Vdiv_w)
        for ki, kw in zip(order, wts):
            ev += kw * F_w[ki]
        ev = ev / (np.linalg.norm(ev) + 1e-12)
        improved[m["mid"]]["lsq_whitened"] = {
            "concepts": kept, "weights": wts,
            "decomp_error": err, "vec": ev.tolist(),
        }
        if (i + 1) % 10 == 0:
            print(f"  whitened {i+1}/{len(manifest)}", flush=True)

    print("loading LPIPS...", flush=True)
    lpips_fn = make_lpips_fn()

    # Restrict to models that have improved decompositions
    M = [m for m in manifest if m["mid"] in improved]
    M_single = [m for m in M if m.get("n_aspects", 1) == 1]

    # Define columns
    columns = []
    for K, levels in [(3, [25,100,400]), (5,[25,75,100,200,400]),
                      (8,[25,50,75,100,150,200,300,400])]:
        sub = [m for m in M_single if m["level"] in levels]
        columns.append((f"single, {K} levels", sub))
    for k in [2, 3, 4]:
        sub = [m for m in M if m.get("n_aspects", 1) == k] + M_single
        columns.append((f"{k}-aspect (+ single pool)", sub))

    variants = ["lsq_orig", "lsq_centered", "nnls_orig", "nnls_centered", "lsq_whitened"]
    results = {}
    for label, sub in columns:
        results[label] = {}
        for v in variants:
            vecs = {m["mid"]: improved[m["mid"]][v]["vec"] for m in sub}
            ss, sm, lp = run_selection(sub, vecs, lpips_fn)
            results[label][v] = {"same_style": ss, "same_mix": sm, "lpips": lp}
            print(f"  [{label}]  {v:14s}  same-style={ss:.2f}  "
                  f"same-mix={sm:.2f}  lpips={lp:.3f}", flush=True)
        print()

    with open("/home/haoming/finexl/runs/results/improvements.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
