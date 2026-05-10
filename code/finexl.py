"""
Minimal reimplementation of FineXL (ECCV 2026 #6486) core method.

Implements Algorithm 1 of the paper:
  1. Vdiv: distributional-divergence vector in CLIP image-embedding space.
  2. Concept-to-vector mapping f(C) via CLIP text encoder using Eq. (6).
  3. Orthogonality filter via Eq. (9).
  4. Linear decomposition Vdiv ~= sum_i w_i * f(C_i) via least-squares,
     with greedy concept addition until decomposition error < e_decomp.

This file deliberately decouples the divergence computation from the
image-generation back-end so the math can be verified end-to-end on a
tiny synthetic case (see finexl_sanity.py) without running a diffusion model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import open_clip


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------

class CLIPEncoder:
    """Thin wrapper over an open_clip CLIP model exposing image/text encoding."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai",
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        toks = self.tokenizer(list(texts)).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def encode_images(self, pil_images) -> np.ndarray:
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Core FineXL operations
# ---------------------------------------------------------------------------

def divergence_vector_from_features(
    base_feats: np.ndarray, personal_feats: np.ndarray
) -> np.ndarray:
    """Eq. (5): Vdiv = mean over prompts of (Enc(G_personal) - Enc(G_base)).

    base_feats and personal_feats are paired (same prompts), shape (n, d)."""
    assert base_feats.shape == personal_feats.shape
    return (personal_feats - base_feats).mean(axis=0)


def concept_to_vector(
    encoder: CLIPEncoder, concept: str, prompts: Sequence[str]
) -> np.ndarray:
    """Eq. (6): f(C) = sum_i (Enc_text(t_i u C) - Enc_text(t_i)).

    We use mean instead of sum so the magnitude is comparable across n.
    Only the direction matters for orthogonality and decomposition."""
    base_emb = encoder.encode_text(list(prompts))
    joined = [_join_prompt_concept(p, concept) for p in prompts]
    personal_emb = encoder.encode_text(joined)
    return (personal_emb - base_emb).mean(axis=0)


def _join_prompt_concept(prompt: str, concept: str) -> str:
    p = prompt.rstrip(" .")
    return f"{p}, {concept} style"


def orthogonality_score(v: np.ndarray, others: Sequence[np.ndarray]) -> float:
    """Eq. (9): signed sum of cos(v, v_j) over already-selected concepts.
    Magnitude indicates redundancy: 0 = orthogonal, large = aligned (or anti-aligned)."""
    if not others:
        return 0.0
    nv = v / (np.linalg.norm(v) + 1e-12)
    s = 0.0
    for u in others:
        nu = u / (np.linalg.norm(u) + 1e-12)
        s += float(np.dot(nv, nu))
    return abs(s)  # we threshold the magnitude, per "total projection exceeds threshold"


def decompose(Vdiv: np.ndarray, concept_vecs: Sequence[np.ndarray]):
    """Solve min_w ||Vdiv - sum_i w_i f(C_i)||_2 via least squares (Eq. 10)."""
    F = np.stack(concept_vecs, axis=1)  # (d, k)
    w, *_ = np.linalg.lstsq(F, Vdiv, rcond=None)
    residual = Vdiv - F @ w
    err = float(np.linalg.norm(residual) / (np.linalg.norm(Vdiv) + 1e-12))
    return w, err


@dataclass
class FineXLResult:
    concepts: list[str]
    weights: list[float]
    decomp_error: float
    rejected_concepts: list[tuple[str, float]]  # (name, orthogonality_score)


def run_finexl(
    Vdiv: np.ndarray,
    candidate_concepts: Sequence[str],
    encoder: CLIPEncoder,
    probe_prompts: Sequence[str],
    e_ortho: float = 0.10,
    e_decomp: float = 0.20,
    verbose: bool = False,
) -> FineXLResult:
    """Algorithm 1 — orthogonality filter + greedy decomposition.

    Returns the kept concepts (in selection order), their weights, and the
    final relative decomposition error.
    """
    kept_names: list[str] = []
    kept_vecs: list[np.ndarray] = []
    rejected: list[tuple[str, float]] = []

    # 1. Pre-compute concept vectors
    concept_vecs = {c: concept_to_vector(encoder, c, probe_prompts)
                    for c in candidate_concepts}

    # 2. Order candidates by alignment with Vdiv (greedy)
    order = sorted(
        candidate_concepts,
        key=lambda c: -abs(_cos(concept_vecs[c], Vdiv)),
    )

    last_err = 1.0
    for c in order:
        v = concept_vecs[c]
        score = orthogonality_score(v, kept_vecs)
        if score >= e_ortho:
            rejected.append((c, score))
            if verbose:
                print(f"  reject {c!r}: ortho={score:.3f} >= {e_ortho}")
            continue
        kept_names.append(c)
        kept_vecs.append(v)
        _, last_err = decompose(Vdiv, kept_vecs)
        if verbose:
            print(f"  keep   {c!r}: ortho={score:.3f}, err={last_err:.3f}")
        if last_err < e_decomp:
            break

    weights, err = decompose(Vdiv, kept_vecs) if kept_vecs else (np.array([]), 1.0)
    return FineXLResult(
        concepts=kept_names,
        weights=[float(w) for w in weights],
        decomp_error=err,
        rejected_concepts=rejected,
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
