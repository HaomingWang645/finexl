"""
Sweep of additional FineXL improvements (variants A–E) on the existing
43-model benchmark. Holds: same V_div from CLIP image features, same
LPIPS scoring, same selection task. Varies: candidate concept pool,
concept-vector geometry, and decomposition.

  A = per-model VLM proposals (from Naive cache) + NNLS + centering
  B = expanded 150-word fixed vocab + NNLS + centering
  C = PCA-truncated whitening (top-K PCs) + NNLS + centering
  D = image-domain concept directions for top-K concepts + NNLS + centering
  E = pool union of B and per-model proposals + NNLS + centering
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
from finexl_improved import improved_decompose, decompose_lsq, decompose_nnls


# ---------- 50-word vocab from earlier (B's seed) ----------
STYLE_VOCAB_50 = [
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
# ---------- expanded 150-word vocab (B) ----------
STYLE_VOCAB_150 = STYLE_VOCAB_50 + [
    # more ground-truth related (all 4 styles, more variants)
    "japonisme", "edo period", "kabuki print", "katsushika",
    "aquarelle", "gouache", "ink wash", "loose brushwork",
    "voxel art", "8bit-pixel", "minecraft style", "low resolution",
    "old master", "baroque painting", "chiaroscuro", "fresco",
    # broader style descriptors
    "expressionist", "fauvist", "cubist", "impressionist",
    "post-impressionist", "neo-impressionist", "abstract expressionism",
    "pointillism", "realism", "hyperrealism", "minimalism",
    "naturalism", "primitive art", "folk art", "outsider art",
    "ukrainian icon", "byzantine", "gothic", "art nouveau",
    "constructivism", "suprematism", "de stijl", "bauhaus",
    "futurism", "vorticism", "dada", "surrealism",
    "magic realism", "color field", "minimalist design",
    "psychedelic art", "vaporwave", "outrun", "synthwave",
    # technical / texture
    "halftone", "stippling", "engraving", "etching",
    "lithograph", "linocut", "screen print", "block print",
    "pen and ink", "marker drawing", "crayon", "pastel chalk",
    "smooth shading", "painterly", "thick brushstrokes", "thin strokes",
    "rough texture", "smooth texture", "grain", "film grain",
    # color palette
    "warm tones", "cool tones", "earth tones", "saturated colors",
    "desaturated", "high saturation", "low saturation",
    "complementary colors", "analogous colors", "triadic palette",
    "limited palette", "rich colors", "faded colors",
    # composition / mood
    "atmospheric", "moody", "ethereal", "whimsical",
    "playful", "serene", "dramatic", "intimate",
    "epic", "ominous", "joyful", "melancholic",
    # rendering style
    "lineart", "outlined", "thick outlines", "thin lines",
    "no outlines", "soft edges", "hard edges 2",
    "high detail", "low detail", "simplified", "complex",
]


def load_manifest():
    with open("/home/haoming/finexl/runs/eval/manifest.json") as f:
        return json.load(f)


def load_explanations():
    out = {}
    for p in [
        "/home/haoming/finexl/runs/results/single/explanations.json",
        "/home/haoming/finexl/runs/results/multi/explanations.json",
    ]:
        with open(p) as f:
            out.update(json.load(f))
    return out


def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


def run_selection(models, vec_per_model, lpips_fn):
    same_style_match = same_mix_match = 0
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
    return same_style_match / n, same_mix_match / n, lpips_sum / n


def make_lpips_fn():
    import lpips as lpips_pkg
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


def variant_decompose(Vdiv, concept_vecs, *, use_centering=True, use_nnls=True,
                      whitening=None, keep_top_k=8):
    """One unified decomposition step. whitening: optional W matrix to apply."""
    F = np.stack(concept_vecs, axis=0)  # (k, d)
    # Optional whitening (applied to f and V_div uniformly)
    if whitening is not None:
        W = whitening
        F = (W @ F.T).T
        Vdiv = W @ Vdiv
    # Optional centering (subtract centroid direction)
    if use_centering:
        mu = F.mean(axis=0)
        n_mu = np.linalg.norm(mu)
        if n_mu > 1e-12:
            d = mu / n_mu
            F = F - np.outer(F @ d, d)
            Vdiv = Vdiv - float(Vdiv @ d) * d
    # Decompose
    if use_nnls:
        from scipy.optimize import nnls
        w, _ = nnls(F.T, Vdiv, maxiter=400)
    else:
        w, *_ = np.linalg.lstsq(F.T, Vdiv, rcond=None)
    err = float(np.linalg.norm(Vdiv - F.T @ w) / (np.linalg.norm(Vdiv) + 1e-12))
    order = np.argsort(-np.abs(w))[:keep_top_k]
    kept_w = w[order]
    expl = (F[order].T @ kept_w)
    n = np.linalg.norm(expl)
    if n > 0:
        expl = expl / n
    return order.tolist(), kept_w.tolist(), err, expl


def build_pca_whitener(F, n_pcs=30, eps=1e-3):
    """Project to top-n_pcs and whiten. Returns matrix W of shape (d, d)."""
    Fc = F - F.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    n_pcs = min(n_pcs, len(S))
    Vk = Vt[:n_pcs]            # (n_pcs, d) right singular vectors
    Sk = S[:n_pcs]             # (n_pcs,)
    # Whitening matrix: project onto top PCs and divide by sqrt(eigenvalue)
    # so that whitened covariance is identity in the n_pcs-dim subspace.
    # Then map back to d-dim by Vk^T (but un-whitened directions are zeroed).
    n_train = max(F.shape[0] - 1, 1)
    sigmas = Sk / np.sqrt(n_train)
    W_pca = (Vk.T * (1.0 / (sigmas + eps))) @ Vk    # (d, d) symmetric
    return W_pca


def build_image_domain_directions(enc, sd_pipe, concepts, k=4, prompts=None):
    """T3: compute concept directions in image space.

    For each concept C, generate k images with prompt "{p}, {C}" using base SD,
    encode them, take mean. Return f_img(C) = mean(Enc_img(I_personal)) - mean(Enc_img(I_base)).

    NOTE: requires SD pipeline; expensive (k * |concepts| generations).
    """
    if prompts is None:
        prompts = PROBE_PROMPTS[:k]
    # Pre-compute base images once
    print(f"  generating base images for {k} prompts...", flush=True)
    base_imgs = []
    for i, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(7000 + i)
        img = sd_pipe(p, num_inference_steps=20, height=512, width=512,
                      generator=g).images[0]
        base_imgs.append(img)
    base_f = enc.encode_images(base_imgs).mean(axis=0)

    out = {}
    for j, c in enumerate(concepts):
        imgs = []
        for i, p in enumerate(prompts):
            g = torch.Generator("cuda").manual_seed(7000 + i + 100 * j)
            img = sd_pipe(f"{p}, {c} style", num_inference_steps=20, height=512,
                          width=512, generator=g).images[0]
            imgs.append(img)
        cf = enc.encode_images(imgs).mean(axis=0)
        out[c] = cf - base_f
    return out


# ----------------- main -----------------

def main():
    manifest = load_manifest()
    explanations = load_explanations()

    print("loading CLIP...", flush=True)
    enc = CLIPEncoder("ViT-B-32", "openai")

    # Pre-compute V_div for every model once (CLIP image features)
    print("computing V_div for all models...", flush=True)
    base_paths = sorted(glob("/home/haoming/finexl/runs/eval/base/*.png"))[:25]
    base_imgs = [Image.open(p).convert("RGB") for p in base_paths]
    base_f = enc.encode_images(base_imgs)
    Vdiv_per_model = {}
    for m in manifest:
        pers_paths = sorted(glob(os.path.join(m["eval_dir"], "*.png")))[:25]
        pers_imgs = [Image.open(p).convert("RGB") for p in pers_paths]
        pers_f = enc.encode_images(pers_imgs)
        Vdiv_per_model[m["mid"]] = divergence_vector_from_features(base_f, pers_f)

    # Pre-compute f(C) for both vocabs
    print(f"computing f(C) for {len(STYLE_VOCAB_150)} concepts (covers both vocabs)...",
          flush=True)
    fC_all = {c: concept_to_vector(enc, c, PROBE_PROMPTS) for c in STYLE_VOCAB_150}

    # PCA whitener from V150
    F150 = np.stack([fC_all[c] for c in STYLE_VOCAB_150], axis=0)
    W_pca = build_pca_whitener(F150, n_pcs=30, eps=1e-3)

    # Helper to get per-model VLM proposals from cached Naive output
    def naive_concepts_for(mid):
        em = explanations.get(mid, {})
        return em.get("Naive", {}).get("concepts", [])

    # Compile per-model explanation vectors for each variant
    print("running variants A–E ...", flush=True)
    vecs_per_variant = {v: {} for v in ["A", "B", "C", "E", "B_lsq", "B_pca_nnls",
                                         "B_pca_lsq"]}
    for m in manifest:
        mid = m["mid"]
        Vdiv = Vdiv_per_model[mid]

        # A: per-model proposals (Naive cache) + NNLS + centering
        a_concepts = naive_concepts_for(mid)
        a_concepts = [c for c in a_concepts if c in fC_all]  # restrict to known
        # Augment with all 50-word vocab if Naive proposals are too small (< 4)
        if len(a_concepts) < 4:
            a_concepts = list(set(a_concepts + STYLE_VOCAB_50))
        a_cv = [fC_all[c] for c in a_concepts]
        order, w, err, ev = variant_decompose(
            Vdiv, a_cv, use_centering=True, use_nnls=True)
        vecs_per_variant["A"][mid] = ev.tolist()

        # B: 150-vocab + NNLS + centering
        b_cv = [fC_all[c] for c in STYLE_VOCAB_150]
        order, w, err, ev = variant_decompose(
            Vdiv, b_cv, use_centering=True, use_nnls=True)
        vecs_per_variant["B"][mid] = ev.tolist()

        # B_lsq: 150-vocab + LSQ + centering (sanity baseline for B)
        order, w, err, ev = variant_decompose(
            Vdiv, b_cv, use_centering=True, use_nnls=False)
        vecs_per_variant["B_lsq"][mid] = ev.tolist()

        # C: PCA-truncated whitening on 150-vocab, then NNLS + centering
        order, w, err, ev = variant_decompose(
            Vdiv, b_cv, use_centering=True, use_nnls=True, whitening=W_pca)
        vecs_per_variant["C"][mid] = ev.tolist()

        # B_pca_nnls: PCA-whitened, NNLS, NO centering (since whitening already centers)
        order, w, err, ev = variant_decompose(
            Vdiv, b_cv, use_centering=False, use_nnls=True, whitening=W_pca)
        vecs_per_variant["B_pca_nnls"][mid] = ev.tolist()

        # B_pca_lsq: PCA-whitened, LSQ, NO centering
        order, w, err, ev = variant_decompose(
            Vdiv, b_cv, use_centering=False, use_nnls=False, whitening=W_pca)
        vecs_per_variant["B_pca_lsq"][mid] = ev.tolist()

        # E: union of (50-vocab) + (per-model Naive proposals) + NNLS + centering
        e_concepts = list(set(STYLE_VOCAB_50 + naive_concepts_for(mid)))
        e_concepts = [c for c in e_concepts if c in fC_all]
        e_cv = [fC_all[c] for c in e_concepts]
        order, w, err, ev = variant_decompose(
            Vdiv, e_cv, use_centering=True, use_nnls=True)
        vecs_per_variant["E"][mid] = ev.tolist()

    # Selection task
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
        for v_name, vec_dict in vecs_per_variant.items():
            ss, sm, lp = run_selection(sub, vec_dict, lpips_fn)
            results[label][v_name] = {"same_style": ss, "same_mix": sm, "lpips": lp}
            print(f"  [{label}]  {v_name:12s}  same-style={ss:.2f}  "
                  f"same-mix={sm:.2f}  lpips={lp:.3f}", flush=True)
        print()

    with open("/home/haoming/finexl/runs/results/iterate.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
