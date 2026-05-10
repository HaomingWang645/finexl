"""
Three baseline explainers, all sharing Qwen2.5-VL as the underlying VLM
(the FineXL paper used GPT-5; we use the same Qwen for all baselines so
comparisons remain fair). Each baseline produces a list of (concept,
weight) pairs that we can feed into the same model-selection metric.

  - Naive: ask the VLM directly for "a list of aspects with scores 0-1".
  - GSCLIP:  one-shot summarization across multiple image pairs, then GPT-style
             re-rank by how distinctive each concept is.
  - VisDiff:  VLM proposes candidate descriptions, then a CLIP-based scorer
              re-ranks by how much each candidate separates personalized vs
              base images. We use the same CLIP encoder as FineXL.

These are simplified faithful re-implementations of the published methods,
not exact code releases (which would require their original repos and
their specific VLM checkpoints).
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from finexl import CLIPEncoder  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Naive baseline: ask VLM directly for aspects with scores
# ---------------------------------------------------------------------------

NAIVE_PROMPT = (
    "These two images are from a base model and a personalized one. "
    "Compare Image 1 (personalized) against Image 2 (base) and identify the "
    "main visual differences. Return a JSON list with up to 6 items, each: "
    "{\"concept\": <single-word adjective or short phrase>, "
    "\"score\": <number 0..1 indicating how strongly this concept differs>}. "
    "Reply with JSON only, no prose."
)


def naive_explanations(qwen, pairs: Sequence[Tuple[Image.Image, Image.Image]],
                       max_pairs: int = 6) -> List[Tuple[str, float]]:
    """Run Naive baseline on multiple pairs, return aggregated (concept, score)."""
    agg: dict[str, list[float]] = {}
    for ip, ib in pairs[:max_pairs]:
        items = _ask_qwen_json(qwen, ip, ib, NAIVE_PROMPT)
        for it in items:
            c = (it.get("concept") or "").strip().lower()
            s = float(it.get("score", 0))
            if c:
                agg.setdefault(c, []).append(max(0.0, min(1.0, s)))
    return sorted([(c, float(np.mean(v))) for c, v in agg.items()],
                  key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# 2. GSCLIP-style baseline (Zhu et al. 2022)
#    Idea: VLM summarizes differences across many pairs; then for each
#    candidate phrase, score = mean cosine(CLIP_text(phrase), V_div).
# ---------------------------------------------------------------------------

GSCLIP_PROMPT = (
    "Compare image 1 (personalized) and image 2 (base). List 6 short phrases "
    "(each <=4 words) describing how Image 1 differs from Image 2 in style, "
    "color, texture, mood, or composition. Comma-separated, no scores."
)


def gsclip_explanations(qwen, pairs, clip_enc: CLIPEncoder,
                        Vdiv: np.ndarray, max_pairs: int = 6
                        ) -> List[Tuple[str, float]]:
    cand: dict[str, int] = {}
    for ip, ib in pairs[:max_pairs]:
        for c in _ask_qwen_csv(qwen, ip, ib, GSCLIP_PROMPT):
            cand[c] = cand.get(c, 0) + 1
    if not cand:
        return []
    phrases = list(cand.keys())
    embs = clip_enc.encode_text(phrases)  # (k, d)
    Vn = Vdiv / (np.linalg.norm(Vdiv) + 1e-12)
    En = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    scores = (En @ Vn).clip(-1, 1)
    out = sorted(zip(phrases, [float(s) for s in scores]), key=lambda kv: -kv[1])
    # rescale scores to [0,1] for downstream consistency with Naive/FineXL
    if out:
        m = max(s for _, s in out)
        if m > 0:
            out = [(p, s / m) for p, s in out]
    return out


# ---------------------------------------------------------------------------
# 3. VisDiff-style baseline (Dunlap et al. 2024)
#    Idea: VLM proposes candidate descriptions; for each, separability =
#    mean cos(CLIP_text(phrase), CLIP_img(personalized)) -
#    mean cos(CLIP_text(phrase), CLIP_img(base)). Sort by separability.
# ---------------------------------------------------------------------------

VISDIFF_PROMPT = (
    "Look at multiple pairs (image 1 = personalized model, image 2 = base "
    "model). Suggest 8 short hypotheses about what differs in the personalized "
    "set, each as a single phrase <=4 words. Comma-separated, no scores."
)


def visdiff_explanations(qwen, pairs, clip_enc: CLIPEncoder,
                         personal_imgs: List[Image.Image],
                         base_imgs: List[Image.Image],
                         max_pairs: int = 6) -> List[Tuple[str, float]]:
    cand: dict[str, int] = {}
    for ip, ib in pairs[:max_pairs]:
        for c in _ask_qwen_csv(qwen, ip, ib, VISDIFF_PROMPT):
            cand[c] = cand.get(c, 0) + 1
    if not cand:
        return []
    phrases = list(cand.keys())
    p_embs = clip_enc.encode_images(personal_imgs)
    b_embs = clip_enc.encode_images(base_imgs)
    t_embs = clip_enc.encode_text(phrases)
    p_n = p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-12)
    b_n = b_embs / (np.linalg.norm(b_embs, axis=1, keepdims=True) + 1e-12)
    t_n = t_embs / (np.linalg.norm(t_embs, axis=1, keepdims=True) + 1e-12)
    sep = (t_n @ p_n.T).mean(axis=1) - (t_n @ b_n.T).mean(axis=1)
    out = sorted(zip(phrases, [float(s) for s in sep]), key=lambda kv: -kv[1])
    if out:
        m = max(s for _, s in out)
        if m > 0:
            out = [(p, s / m) for p, s in out]
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask_qwen_json(qwen, img1: Image.Image, img2: Image.Image, prompt: str):
    raw = qwen._raw_query([img1, img2], prompt, max_new_tokens=256)
    try:
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    # fallback: parse loose lines like "vibrant: 0.7"
    out = []
    for ln in raw.splitlines():
        m = re.match(r"\s*[-*]?\s*['\"]?(\w[\w\s\-]+?)['\"]?\s*[:=]\s*([-0-9\.]+)", ln)
        if m:
            out.append({"concept": m.group(1).strip(), "score": float(m.group(2))})
    return out


def _ask_qwen_csv(qwen, img1: Image.Image, img2: Image.Image, prompt: str
                  ) -> List[str]:
    raw = qwen._raw_query([img1, img2], prompt, max_new_tokens=128)
    items = re.split(r"[,;\n]+", raw)
    out = []
    for it in items:
        s = it.strip().strip(".'\" ")
        s = s.lower()
        if s and 1 <= len(s.split()) <= 5 and len(s) < 40:
            if s not in out:
                out.append(s)
    return out
