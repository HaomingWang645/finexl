"""
Improvements to FineXL based on the reproduction diagnostic.

  T1. Joint-component subtraction. Every f(C) shares a common "appended-text"
      direction in CLIP space (cos(V_div, f(C)) ≈ 0.10 even for unrelated C).
      We subtract this shared direction before decomposition, which sharpens
      the signal-to-distractor ratio.

  T4. Sparse non-negative LASSO. Replace the unconstrained LSQ in Eq. (10)
      with non-negative LASSO. This forces a small number of large positive
      weights, which is the structure FineXL claims to discover.

Both can be combined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List

import numpy as np
from scipy.optimize import nnls


def center_concepts(concept_vecs: List[np.ndarray]):
    """Compute centroid direction μ̂ = μ / ‖μ‖ over the candidate set."""
    M = np.stack(concept_vecs, axis=0)        # (k, d)
    mu = M.mean(axis=0)                       # (d,)
    n_mu = np.linalg.norm(mu)
    if n_mu < 1e-12:
        return concept_vecs, np.zeros_like(mu)
    direction = mu / n_mu
    return direction


def project_out(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the projection of v onto a unit direction."""
    return v - float(np.dot(v, direction)) * direction


def decompose_lsq(Vdiv: np.ndarray, concept_vecs: Sequence[np.ndarray]):
    F = np.stack(concept_vecs, axis=1)
    w, *_ = np.linalg.lstsq(F, Vdiv, rcond=None)
    err = float(np.linalg.norm(Vdiv - F @ w) / (np.linalg.norm(Vdiv) + 1e-12))
    return w, err


def decompose_nnls(Vdiv: np.ndarray, concept_vecs: Sequence[np.ndarray]):
    """T4: non-negative LSQ. Forces w_i >= 0; scipy.optimize.nnls."""
    F = np.stack(concept_vecs, axis=1)
    # NNLS requires F shape (m, n). m=d (CLIP dim), n=k.
    # If F has too many columns, NNLS still works but tends to give sparse w.
    w, _residual = nnls(F, Vdiv, maxiter=200)
    err = float(np.linalg.norm(Vdiv - F @ w) / (np.linalg.norm(Vdiv) + 1e-12))
    return w, err


@dataclass
class ImprovedResult:
    concepts: List[str]
    weights: List[float]
    decomp_error: float
    explanation_vec: np.ndarray  # what to use for selection-task cosine


def improved_decompose(
    Vdiv: np.ndarray,
    concept_names: Sequence[str],
    concept_vecs: Sequence[np.ndarray],
    *,
    use_centering: bool = True,
    use_nnls: bool = False,
    keep_top_k: int = 8,
) -> ImprovedResult:
    """Center, decompose, return top-K concepts + a centered explanation vector.

    Returns:
      concepts/weights: top-K by |w|.
      decomp_error: relative residual on the centered V_div.
      explanation_vec: sum_i w_i * f̃(C_i), a centered representation that we
        use directly for the model-selection cosine.
    """
    if use_centering:
        direction = center_concepts(list(concept_vecs))
        if np.linalg.norm(direction) > 0:
            cv_centered = [project_out(v, direction) for v in concept_vecs]
            Vdiv_c = project_out(Vdiv, direction)
        else:
            cv_centered = list(concept_vecs)
            Vdiv_c = Vdiv
    else:
        cv_centered = list(concept_vecs)
        Vdiv_c = Vdiv

    if use_nnls:
        w, err = decompose_nnls(Vdiv_c, cv_centered)
    else:
        w, err = decompose_lsq(Vdiv_c, cv_centered)

    # Top-K by |w|
    order = np.argsort(-np.abs(w))[:keep_top_k]
    kept_names = [concept_names[i] for i in order]
    kept_weights = [float(w[i]) for i in order]
    kept_vecs = [cv_centered[i] for i in order]

    expl_vec = np.zeros_like(Vdiv_c)
    for wi, vi in zip(kept_weights, kept_vecs):
        expl_vec += wi * vi
    n = np.linalg.norm(expl_vec)
    if n > 0:
        expl_vec = expl_vec / n

    return ImprovedResult(kept_names, kept_weights, err, expl_vec)
