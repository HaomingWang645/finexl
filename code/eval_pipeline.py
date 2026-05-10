"""
End-to-end evaluation pipeline (Sections 5 + 6.1 of the paper).

For every personalized model M in the pool:
  1. Build an explanation vector e(M) per method (FineXL + 3 baselines).
For every target M_t:
  2. Pick the candidate M_c (M_c != M_t) whose explanation is closest.
  3. Score the selection by mean LPIPS between target and candidate images
     under matched prompts. Lower is better.

We also report selection accuracy: fraction of targets where the selected
candidate is the *level-matched* same-style model.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from finexl import (  # noqa
    CLIPEncoder, divergence_vector_from_features, run_finexl, concept_to_vector,
)
from finexl_full import QwenConceptProposer, PROBE_PROMPTS  # noqa
from baselines import (  # noqa
    naive_explanations, gsclip_explanations, visdiff_explanations,
)


# ---------------------------------------------------------------------------
# Explanation -> vector representation
# ---------------------------------------------------------------------------

class ConceptVecCache:
    """Cache f(C) (Eq. 6) for re-use across methods."""

    def __init__(self, clip_enc: CLIPEncoder, probe_prompts):
        self.enc = clip_enc
        self.probes = probe_prompts
        self._cache: dict = {}

    def f(self, c: str) -> np.ndarray:
        if c not in self._cache:
            self._cache[c] = concept_to_vector(self.enc, c, self.probes)
        return self._cache[c]


def explanation_to_vector(concepts: List[str], weights: List[float],
                          fcache: ConceptVecCache) -> np.ndarray:
    """sum_i w_i * f(C_i) per Eq. 6 -- the same f used inside FineXL.

    Same probe-prompt set across methods, so distances are comparable.
    """
    if not concepts:
        return np.zeros((512,), dtype=np.float64)
    v = np.zeros_like(fcache.f(concepts[0]))
    for c, w in zip(concepts, weights):
        v += float(w) * fcache.f(c)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


# ---------------------------------------------------------------------------
# Model pool: each "model" is a folder of evaluation images + metadata
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, mid: str, eval_dir: str, style: str, level: int,
                 mix: Dict[str, float] | None = None, **extra):
        self.mid = mid
        self.eval_dir = eval_dir
        self.style = style
        self.level = level
        self.mix = mix or {style: 1.0}
        self.n_aspects = extra.get("n_aspects", len(self.mix))

    def load_images(self, n: int | None = None) -> List[Image.Image]:
        paths = sorted(glob(os.path.join(self.eval_dir, "*.png")))
        if n is not None:
            paths = paths[:n]
        return [Image.open(p).convert("RGB") for p in paths]


# ---------------------------------------------------------------------------
# Metric: pairwise LPIPS averaged over matching prompts
# ---------------------------------------------------------------------------

class LPIPSScorer:
    def __init__(self, device="cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, A: List[Image.Image], B: List[Image.Image]) -> float:
        """Mean LPIPS between paired images A[i], B[i] in [0, 1]."""
        from torchvision import transforms as T
        tx = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(256),
            T.ToTensor(),
        ])
        n = min(len(A), len(B))
        ds = []
        for i in range(n):
            a = tx(A[i]).unsqueeze(0).to(self.device) * 2 - 1
            b = tx(B[i]).unsqueeze(0).to(self.device) * 2 - 1
            d = self.net(a, b).mean().item()
            ds.append(d)
        return float(np.mean(ds)) if ds else float("nan")


# ---------------------------------------------------------------------------
# Main: explain every model, then run the selection task
# ---------------------------------------------------------------------------

def explain_one_model(M: Model, base_imgs: List[Image.Image],
                      qwen: QwenConceptProposer, clip_enc: CLIPEncoder,
                      fcache: 'ConceptVecCache',
                      e_ortho: float = 0.30, e_decomp: float = 0.20,
                      n_proposer_pairs: int = 4) -> Dict[str, dict]:
    pers_imgs = M.load_images()
    n = min(len(base_imgs), len(pers_imgs))
    base_imgs = base_imgs[:n]
    pers_imgs = pers_imgs[:n]
    pairs = list(zip(pers_imgs, base_imgs))

    base_f = clip_enc.encode_images(base_imgs)
    pers_f = clip_enc.encode_images(pers_imgs)
    Vdiv = divergence_vector_from_features(base_f, pers_f)

    explanations: Dict[str, dict] = {}

    # FineXL
    proposals = qwen.propose_for_pairs(pairs, max_pairs=n_proposer_pairs)
    res = run_finexl(Vdiv, proposals, clip_enc, PROBE_PROMPTS,
                     e_ortho=e_ortho, e_decomp=e_decomp)
    explanations["FineXL"] = {
        "concepts": res.concepts, "weights": res.weights,
        "decomp_error": res.decomp_error,
        "vec": explanation_to_vector(res.concepts, res.weights, fcache).tolist(),
    }

    # Naive
    naive = naive_explanations(qwen, pairs, max_pairs=n_proposer_pairs)
    explanations["Naive"] = {
        "concepts": [c for c, _ in naive],
        "weights": [w for _, w in naive],
        "vec": explanation_to_vector([c for c, _ in naive],
                                     [w for _, w in naive], fcache).tolist(),
    }

    # GSCLIP
    gs = gsclip_explanations(qwen, pairs, clip_enc, Vdiv,
                             max_pairs=n_proposer_pairs)
    explanations["GSCLIP"] = {
        "concepts": [c for c, _ in gs],
        "weights": [w for _, w in gs],
        "vec": explanation_to_vector([c for c, _ in gs],
                                     [w for _, w in gs], fcache).tolist(),
    }

    # VisDiff
    vd = visdiff_explanations(qwen, pairs, clip_enc, pers_imgs, base_imgs,
                              max_pairs=n_proposer_pairs)
    explanations["VisDiff"] = {
        "concepts": [c for c, _ in vd],
        "weights": [w for _, w in vd],
        "vec": explanation_to_vector([c for c, _ in vd],
                                     [w for _, w in vd], fcache).tolist(),
    }

    return explanations


def run_selection_task(models: List[Model], explanations: Dict[str, Dict[str, dict]],
                       lpips_scorer: LPIPSScorer, method: str
                       ) -> Tuple[float, float, List[dict]]:
    """For each target, pick the closest non-self candidate by explanation
    cosine similarity. Return (mean LPIPS to selected, exact-match accuracy)."""
    rows = []
    lpips_sum = 0.0
    matches = 0

    # Pre-compute vectors
    vecs = {}
    for M in models:
        v = np.asarray(explanations[M.mid][method]["vec"], dtype=np.float64)
        vecs[M.mid] = v / (np.linalg.norm(v) + 1e-12)

    for tgt in models:
        tv = vecs[tgt.mid]
        best, best_sim = None, -2.0
        for cand in models:
            if cand.mid == tgt.mid:
                continue
            sim = float(np.dot(tv, vecs[cand.mid]))
            if sim > best_sim:
                best, best_sim = cand, sim
        # LPIPS between target and selected images
        tgt_imgs = tgt.load_images()
        cand_imgs = best.load_images()
        d = lpips_scorer.score(tgt_imgs, cand_imgs)
        lpips_sum += d
        # exact match: same style and same level
        ok = (best.style == tgt.style) and (best.level == tgt.level
                                            or abs(best.level - tgt.level) <= 1)
        matches += int(ok)
        rows.append({"target": tgt.mid, "selected": best.mid,
                     "sim": best_sim, "lpips": d, "match": ok})
    return lpips_sum / len(models), matches / len(models), rows


def random_baseline(models: List[Model], lpips_scorer: LPIPSScorer,
                    seed: int = 0, n_repeats: int = 5) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_repeats):
        for tgt in models:
            others = [m for m in models if m.mid != tgt.mid]
            cand = others[int(rng.integers(0, len(others)))]
            d = lpips_scorer.score(tgt.load_images(), cand.load_images())
            vals.append(d)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON describing models: list of {mid, eval_dir, style, level, mix?}")
    ap.add_argument("--base_eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--explanations_only", action="store_true")
    ap.add_argument("--levels_subset", type=int, nargs="+", default=None,
                    help="optional: restrict to a subset of levels (e.g. 25 100 400)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.manifest) as f:
        manifest = json.load(f)
    models = [Model(**m) for m in manifest]
    if args.levels_subset is not None:
        models = [m for m in models if m.level in args.levels_subset]
    print(f"pool size: {len(models)}", flush=True)

    print("loading CLIP...", flush=True)
    clip_enc = CLIPEncoder("ViT-B-32", "openai")
    fcache = ConceptVecCache(clip_enc, PROBE_PROMPTS)

    # Explanations
    expl_path = os.path.join(args.out_dir, "explanations.json")
    if os.path.exists(expl_path):
        print(f"using cached {expl_path}", flush=True)
        with open(expl_path) as f:
            explanations = json.load(f)
    else:
        print("loading Qwen2.5-VL...", flush=True)
        qwen = QwenConceptProposer()
        base_imgs = [Image.open(p).convert("RGB")
                     for p in sorted(glob(os.path.join(args.base_eval_dir, "*.png")))]
        explanations = {}
        for i, M in enumerate(models):
            print(f"[{i+1}/{len(models)}] explain {M.mid} ...", flush=True)
            explanations[M.mid] = explain_one_model(
                M, base_imgs, qwen, clip_enc, fcache)
            with open(expl_path, "w") as f:
                json.dump(explanations, f, indent=2)
        del qwen
        gc.collect()
        torch.cuda.empty_cache()

    if args.explanations_only:
        return

    print("loading LPIPS...", flush=True)
    lpips_scorer = LPIPSScorer()

    results = {}
    for method in ["FineXL", "Naive", "GSCLIP", "VisDiff"]:
        m_lpips, m_acc, rows = run_selection_task(
            models, explanations, lpips_scorer, method)
        results[method] = {"lpips": m_lpips, "acc": m_acc, "rows": rows}
        print(f"{method:8s}  lpips={m_lpips:.3f}  acc={m_acc:.3f}", flush=True)

    rnd = random_baseline(models, lpips_scorer)
    results["Random"] = {"lpips": rnd}
    print(f"{'Random':8s}  lpips={rnd:.3f}", flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
