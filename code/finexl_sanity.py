"""
Tiny synthetic sanity check for the FineXL math.

Setup
-----
We avoid running a diffusion model. Instead, since CLIP image and text
encoders share an aligned space, we construct a *synthetic* divergence
vector V_div as the mean text-embedding difference between

    "<COCO caption>"
and
    "<COCO caption>, <ground-truth style words> style"

with two ground-truth styles ("vibrant" and "abstract"). FineXL should then
recover those two concepts and assign them the largest weights, while
rejecting near-synonyms via orthogonality and assigning low weights to
unrelated distractors.

This doesn't replace a real image-domain experiment, but it directly
verifies that the orthogonality filter + linear decomposition recover
the planted concepts in the correct embedding space.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from finexl import (
    CLIPEncoder, run_finexl, divergence_vector_from_features,
    _join_prompt_concept,
)


# 25 short COCO-flavored prompts (manually written, no API needed)
PROBE_PROMPTS = [
    "a man with a red helmet on a small moped on a dirt road",
    "a kitchen with stainless steel appliances and wood floors",
    "two giraffes standing in a grassy field at sunset",
    "a boy throwing a frisbee to a dog on a beach",
    "a wooden bench in front of a brick wall",
    "a plate with a sandwich and french fries",
    "a young woman holding an umbrella in the rain",
    "a busy city street with yellow taxis",
    "a black cat sleeping on a couch",
    "a snowboarder jumping in the air on a mountain",
    "a chef preparing food in a restaurant kitchen",
    "a small sailboat on a calm lake at dawn",
    "a child blowing out birthday candles on a cake",
    "an old library with tall bookshelves",
    "a road winding through autumn forest",
    "two dogs playing tug of war with a rope",
    "a violin resting on sheet music",
    "a young man riding a skateboard down a ramp",
    "a market stall with fresh vegetables",
    "a woman reading a book in a coffee shop",
    "a hot air balloon floating over green fields",
    "a polar bear walking on broken ice",
    "a guitarist performing on a small stage",
    "a row of bicycles parked outside a cafe",
    "a fisherman casting a line at sunrise",
]

# Ground-truth styles planted in the divergence vector.
PLANTED = ["vibrant", "abstract"]

# Candidate concept pool for FineXL to discover from. Includes:
#  - both planted concepts
#  - near-synonyms that should be filtered by the orthogonality test
#  - clearly unrelated distractors that should get small weights
CANDIDATES = [
    # planted
    "vibrant", "abstract",
    # near-synonyms (should be redundant => high ortho score => rejected)
    "vivid", "colorful", "non-figurative", "geometric abstraction",
    # unrelated distractors
    "monochrome", "photorealistic", "vintage", "minimalist", "pixel-art",
    "sketch", "oil painting", "blurry", "high-contrast",
]


def build_planted_Vdiv(encoder: CLIPEncoder, prompts, planted) -> np.ndarray:
    """Construct V_div as the mean (Enc('p, planted... style') - Enc('p'))."""
    base_emb = encoder.encode_text(list(prompts))
    style = " and ".join(planted)
    personal = [_join_prompt_concept(p, style) for p in prompts]
    pers_emb = encoder.encode_text(personal)
    return divergence_vector_from_features(base_emb, pers_emb)


def main():
    print("[sanity] loading CLIP ViT-B-32 (openai)...", flush=True)
    enc = CLIPEncoder(model_name="ViT-B-32", pretrained="openai")
    print(f"[sanity] device={enc.device}", flush=True)

    print(f"[sanity] planted concepts: {PLANTED}")
    Vdiv = build_planted_Vdiv(enc, PROBE_PROMPTS, PLANTED)
    print(f"[sanity] |V_div|={np.linalg.norm(Vdiv):.4f}, "
          f"dim={Vdiv.shape[0]}")

    print("[sanity] running FineXL Algorithm 1 ...")
    res = run_finexl(
        Vdiv=Vdiv,
        candidate_concepts=CANDIDATES,
        encoder=enc,
        probe_prompts=PROBE_PROMPTS,
        e_ortho=0.10,
        e_decomp=0.20,
        verbose=True,
    )

    out = {
        "planted": PLANTED,
        "kept_concepts": res.concepts,
        "kept_weights": res.weights,
        "decomp_error_relative": res.decomp_error,
        "rejected_for_orthogonality": [
            {"concept": c, "ortho_score": s} for c, s in res.rejected_concepts
        ],
    }
    print("\n[sanity] kept concepts and weights:")
    for c, w in zip(res.concepts, res.weights):
        marker = "  <-- planted" if c in PLANTED else ""
        print(f"   {c:25s}  w={w:+.3f}{marker}")
    print(f"\n[sanity] relative decomposition error: {res.decomp_error:.3f}")
    if res.rejected_concepts:
        print("[sanity] rejected for orthogonality:")
        for c, s in res.rejected_concepts:
            print(f"   {c:25s}  ortho_score={s:.3f}")

    # Sanity claim: planted concepts should dominate the kept set with the
    # highest absolute weights.
    kept = dict(zip(res.concepts, res.weights))
    top_two = sorted(kept.items(), key=lambda kv: -abs(kv[1]))[:2]
    print(f"\n[sanity] top-2 concepts by |w|: {[c for c,_ in top_two]}")
    success = all(c in PLANTED or any(p in c or c in p for p in PLANTED)
                  for c, _ in top_two)
    print(f"[sanity] planted concepts recovered in top-2: {success}")

    out["top_two_recovered"] = success

    # ----- Second pass: disable orthogonality filter to verify the linear-
    # decomposition math in isolation (Eq. 10 with all candidates kept).
    print("\n[sanity-B] running with orthogonality filter disabled (e_ortho=inf)")
    res2 = run_finexl(
        Vdiv=Vdiv,
        candidate_concepts=CANDIDATES,
        encoder=enc,
        probe_prompts=PROBE_PROMPTS,
        e_ortho=float("inf"),
        e_decomp=0.0,  # never break early -> use every candidate
        verbose=False,
    )
    print("[sanity-B] kept concepts (sorted by |w|):")
    pairs = sorted(zip(res2.concepts, res2.weights), key=lambda kv: -abs(kv[1]))
    for c, w in pairs:
        marker = "  <-- planted" if c in PLANTED else ""
        print(f"   {c:25s}  w={w:+.3f}{marker}")
    print(f"[sanity-B] relative decomposition error: {res2.decomp_error:.3f}")
    top_two_b = [c for c, _ in pairs[:2]]
    success_b = all(c in PLANTED for c in top_two_b)
    print(f"[sanity-B] top-2 concepts by |w|: {top_two_b}")
    print(f"[sanity-B] planted concepts recovered in top-2: {success_b}")
    out["pass_B_no_ortho_filter"] = {
        "concepts": res2.concepts,
        "weights": res2.weights,
        "decomp_error_relative": res2.decomp_error,
        "top_two": top_two_b,
        "top_two_recovered": success_b,
    }

    os.makedirs("/home/haoming/finexl/results", exist_ok=True)
    with open("/home/haoming/finexl/results/sanity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[sanity] wrote /home/haoming/finexl/results/sanity.json")


if __name__ == "__main__":
    main()
