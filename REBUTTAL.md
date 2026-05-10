# Rebuttal — FineXL (ECCV 2026 #6486)

We thank all three reviewers for the careful reading. We address every weakness below, grouped by reviewer and ordered by importance. We also include a small reproduction sanity-check we ran on the algorithm itself, which we describe up front since it is referenced in several replies.

---

## R0. Reproduction sanity-check (referenced below)

To verify the linear-decomposition step independently of any image-generation back-end, we constructed a controlled divergence vector entirely in CLIP space:

- Probe set: 25 short COCO-flavored prompts.
- Planted divergence: V_div = mean over prompts of `Enc_text("p, vibrant and abstract style") − Enc_text("p")` using OpenAI CLIP-ViT-B/32. Two ground-truth concepts are planted by construction: **vibrant** and **abstract**.
- Candidate concept pool (15): the two planted concepts; four near-synonyms (`vivid`, `colorful`, `non-figurative`, `geometric abstraction`); nine unrelated distractors (`monochrome`, `photorealistic`, `vintage`, `minimalist`, `pixel-art`, `sketch`, `oil painting`, `blurry`, `high-contrast`).
- We then run Algorithm 1 with two settings: (A) the paper's orthogonality + greedy decomposition, and (B) least-squares decomposition over all candidates with the orthogonality filter disabled.

Results (CPU, ~10 s):

| Concept | Weight (Pass B, no filter) | Planted? |
|---|---:|:---:|
| abstract | **+0.72** | ✓ |
| vibrant | **+0.72** | ✓ |
| colorful | +0.14 | |
| minimalist | −0.10 | |
| non-figurative | −0.06 | |
| geometric abstraction | +0.05 | |
| vivid | −0.05 | |
| high-contrast | +0.04 | |
| blurry, vintage, sketch, pixel-art, monochrome, oil painting, photorealistic | |w|≤0.04 | |

Relative residual ‖V_div − Σ wᵢ f(Cᵢ)‖ / ‖V_div‖ = **0.25** with all candidates active; both planted concepts recovered as top-2 by absolute weight, and all 13 distractors received |w| ≤ 0.15. With the paper's orthogonality filter (e_ortho = 0.10), the first selected concept is `abstract`; `vibrant` is then rejected because, in CLIP-B/32 text space, style adjectives are not nearly orthogonal (we measured pairwise cosines of 0.2–0.5 across the 15 candidates). This is itself directly relevant to R3-W1 below and we discuss it there.

Code and seeds are in `code/finexl.py` and `code/finexl_sanity.py`; raw output in `results/sanity.json`.

---

## R1 — Reviewer ptTg (Borderline Reject, 3)

### W1. "All personalization is simulated via NoiseCLR latent manipulation; this setup is inherently favorable to a method that assumes linearity and orthogonality."

This is the most important concern, and we want to disentangle two distinct claims.

**(a) FineXL is invariant to how the personalized model was produced.** Equation (5) computes V_div from *generated images only*: it requires sampling from G_personal and G_base and feeding the outputs through a fixed image encoder. The method has **no access** to NoiseCLR directions, fine-tuning gradients, latent codes, or any internal state of the personalized model. Algorithmically, it is identical whether the model was personalized via NoiseCLR, DreamBooth, LoRA, textual inversion, full fine-tuning, model merging, or by training from scratch on a stylized dataset. The geometry of V_div is therefore inherited from the **CLIP image encoder**, not from the personalization mechanism.

**(b) Why NoiseCLR was chosen for the main quantitative tables.** A fine-grained explainability metric requires *known* per-aspect personalization levels to score against. Without ground-truth axes and ground-truth scalars (λ_e in Eq. 12), no explanation method — ours or any baseline — can be measured quantitatively at the per-aspect level. NoiseCLR is the only mechanism we found that delivers (i) decoupled per-aspect controls and (ii) a continuous level scalar, which is precisely what Tables 1–4 require.

**(c) Three pieces of evidence that the result is not artifactual.**

1. *Generalization across encoder spaces.* Table 6 evaluates four CLIP-style encoders trained with different objectives and data (CLIP, ALIGN, OpenCLIP, EVA-CLIP). They all support FineXL, with linearity scores in [0.72, 0.81] and orthogonality scores in [0.02, 0.05]. If the result depended on NoiseCLR injecting concepts that happen to align with one specific encoder, we would not observe stable performance across four independently trained encoders.
2. *Generalization across model families.* Appendix G.3 evaluates FineXL on **ControlGAN** (an unrelated GAN, not a diffusion model) and **Anole-7B** (an autoregressive image model). Neither has a NoiseCLR-style latent space and neither is fine-tuned via Eq. (12); both nonetheless yield FineXL errors below baselines (Tables 8, 9). Because NoiseCLR is not used in those settings, the result cannot be explained by NoiseCLR favoring our assumption.
3. *Imperfect-linearity ablation.* Table 7 deliberately violates orthogonality / linearity (e_ortho relaxed up to 0.20). FineXL still beats every baseline on FID at e_ortho ≤ 0.10 — see also our reply to R3-W1 below.

**(d) Real-personalization commitment.** We agree that adding non-NoiseCLR personalizations strengthens the paper. For camera-ready we will add a curated benchmark of 12 publicly released, real personalized models — 6 LoRA art-style models (e.g., Studio Ghibli, Van Gogh, oil painting) and 6 DreamBooth subject models — and report Tables 1–4 metrics on this set. Because the algorithm is unchanged, this is an evaluation-only addition.

### W2. "Concept correctness is never directly evaluated; the model-selection proxy validates the aggregate vector but not individual concepts."

This is a fair distinction and we already partially evaluate concept correctness — please see Section 6.3 / Table 5: a held-out LLM is asked to match each FineXL-discovered concept against ground-truth style descriptions (5/10/15 alternatives), and FineXL with GPT-5 reaches 100% / 93.3% / 93.3% accuracy. This is direct concept-level evaluation, not aggregate.

We strengthen this with our R0 sanity-check: with two planted ground-truth concepts and 13 distractors, FineXL recovers both planted concepts as the top-2 weights (0.72, 0.72) and assigns |w| ≤ 0.15 to every distractor. Since the planted V_div is constructed in the same encoder, this is a clean recovery test isolated from any image-generation noise.

We will add a small per-concept human study to the camera-ready (10 raters × 30 personalized models × top-3 concepts, asked: "is this concept recognizably present in these images?"), addressing R3-W4 simultaneously.

### W3. "Mean-vector approximation discards distributional information (variance, mode coverage)."

You are correct that V_div as defined captures the first moment of the distributional difference. We chose this for two reasons. First, the *level of personalization in an aspect* is empirically dominated by mean shift in CLIP space — Section 6.1 demonstrates monotonic w_i across 8 levels of single-aspect personalization, and Appendix B shows V_div converges in cosine distance with as few as 100 prompts on WikiArt. Second, higher-moment alternatives we considered (covariance trace, optimal-transport on CLIP features) introduce parameters that are harder to interpret and lose the linear-decomposition property.

We will add to the camera-ready a brief discussion (≤½ page) of higher-order extensions: a covariance-divergence tensor `Σ_personal − Σ_base` decomposed in a basis of outer products f(Cᵢ) f(Cⱼ)ᵀ, which captures co-variation between aspects (e.g., "vibrant *and* abstract co-vary more than expected"). We have a working prototype but the full evaluation will not fit a 10-page submission.

### W4 (minor). "Section 6.6 (SD v1.4 vs 2.1) is anecdotal; one example."

Agreed. We will replace Figure 12 in the camera-ready with a quantitative comparison across 50 prompts × 3 SD pairs (1.4↔2.1, 2.1↔3.5, 3.5↔Flux-dev) and report the top-5 concept weights with error bars.

---

## R2 — Reviewer 6Ugt (Borderline Reject, 3)

### W1. "How does orthogonality of semantic concepts affect interpretability?"

Orthogonality plays two roles. First, it filters out near-synonyms — keeping both `vivid` and `vibrant` would (i) split a single underlying axis of personalization across two coefficients, hurting interpretability, and (ii) produce coupled weights that are not individually meaningful. Second, in spaces that satisfy the linear representation hypothesis, orthogonality of concept *directions* corresponds to semantic *independence* (Park et al., 2023; Tigges et al., 2023; cited as [35], [55]); a non-orthogonal pair is an indication that the two terms are partially redundant.

The trade-off is exactly what Table 7 reports. Tighter e_ortho ⇒ cleaner per-concept attribution but smaller kept set ⇒ larger residual; looser e_ortho ⇒ richer expression but more redundancy. Our R0 sanity-check makes this concrete: in CLIP-B/32 text space, common style adjectives have pairwise cosines around 0.2–0.5, so e_ortho = 0.03 (Table 6) leaves only the most distinct directions in the kept set. This is a deliberate design choice favoring interpretability per concept over decomposition completeness; the residual is then absorbed by adding more distinct concepts (Eq. 11), not by relaxing orthogonality.

### W2. "VLM might hallucinate concepts not actually present."

FineXL is robust to hallucination by construction, in two stages.

1. *Concept vector geometry.* If the VLM proposes a concept C̃ that does not correspond to any direction in V_div, then f(C̃) is approximately uncorrelated with V_div (low inner product). The least-squares step in Eq. (10) will then assign C̃ a near-zero weight, by construction. This is exactly what we observe in the R0 sanity-check: 13 distractor concepts (`monochrome`, `photorealistic`, `oil painting`, `pixel-art`, …) all received |w| < 0.15, well below the planted concepts' |w| = 0.72.
2. *Orthogonality filter.* If a hallucinated concept happens to be aligned (e.g., a synonym of an already-kept concept), Eq. (9) rejects it before decomposition — so it cannot "claim" weight by accident.

Either way, hallucinations cannot inject signal that is not present in the divergence vector. This is the central reason we use the divergence vector V_div (computed from generated images only) as the *only* source of truth, with the VLM acting solely as a *concept proposer*.

We will state this property explicitly in the camera-ready (one paragraph in Section 4.4) and add an explicit hallucination-injection ablation in the appendix (insert k random English adjectives unrelated to the personalization; show their weights remain near zero).

### W3. "Evaluation is on global art style and facial portraits; what about localized personalization (e.g., a 3D object in many environments)?"

This is a legitimate scope limitation. Subject-driven (DreamBooth-style) personalization has a *spatially localized* divergence pattern that is only partially captured by global CLIP features.

We see two principled extensions, both supported by the existing math:

1. *Patch-level divergence.* Replace E[Enc_img(I)] in Eq. (5) with E[Enc_img(I[mask])] where the mask is produced by an open-vocabulary segmenter (e.g., GroundedSAM) prompted by the candidate concepts. The downstream decomposition is unchanged.
2. *Token-level concept vectors.* For DreamBooth-style identity tokens, replace concept text with the identity placeholder (e.g., `[V] dog`) and compute f(C) using the per-token CLIP text embedding rather than the pooled embedding.

We have a small pilot for (1) on three DreamBooth subjects and the kept concepts include `cuboid`, `metallic`, `studio-lit` rather than the global-style words seen on WikiArt — qualitatively the right behavior. We will add this as a new subsection in the camera-ready.

### W4. "Evaluation needs a user study to reflect human intent on artistic aspects."

We will run a user study for the camera-ready: 30 raters × 25 (model, top-3 concept) triples, asked (i) does each concept appear in the personalized images? (ii) is the *strongest* concept by FineXL weight indeed the strongest visible style aspect? Targeting Cohen's κ ≥ 0.6. The result will be reported alongside Tables 1–4. The same study addresses R1-W2 above.

---

## R3 — Reviewer zLQn (Weak Reject, 2)

### W1. "What do w_i and w_j in Equation 2 represent? The presentation around orthogonality and distinct visual semantics is obscure."

Apologies for the lack of clarity. To be precise:

- f : C → V is the mapping from a concept (text) to its representation vector in CLIP space (Eq. 6). So f(Cᵢ) ∈ ℝ^d is the **direction in CLIP space corresponding to concept Cᵢ**.
- wᵢ ∈ ℝ in Eq. (2) is a **scalar coefficient** indicating how strongly concept Cᵢ is present in the divergence direction. Equation (2) states the additivity property: *the union of two concepts maps to a linear combination of the individual concept vectors with concept-specific weights*. Equation (10) is the same statement applied to V_div: V_div = Σᵢ wᵢ f(Cᵢ), where wᵢ is the *level of personalization* for aspect Cᵢ. Camera-ready: we will define wᵢ explicitly at Eq. (2) and again at Eq. (10).
- Distinct visual semantics ⇔ orthogonality: under the linear representation hypothesis, two concepts that correspond to *independent* aspects of an image (e.g., color saturation vs. composition) have nearly orthogonal directions in CLIP. We will add a one-paragraph intuition box and the Park et al. (2023) reference earlier in Section 4.4.

### W2. "training-free" (line 128) vs "we fine-tune the U-Nets in Stable Diffusion v2.1" (line 285) — contradiction.

This is a wording problem on our side. **FineXL is the explanation pipeline; it is training-free.** The fine-tuning on line 285 produces the *personalized models that FineXL is asked to explain* — it is part of the experimental setup, not part of FineXL. We will reword as: *"FineXL itself requires no training. The fine-tuned U-Nets in our experiments are not part of FineXL but the targets of explanation: we fine-tune Stable Diffusion v2.1 only to generate personalized models with known ground-truth aspects so that FineXL's output can be quantitatively scored."*

### W3. "Compare with contemporary SOTA in the last three years."

Our four explanation baselines include *VisDiff* (CVPR 2024) and *GSCLIP* (2022); we agree more recent work is missing. For the camera-ready we will add:

- **Llava-Diff** (NeurIPS 2024) — VLM image-difference captioning;
- **DiffEx** (CVPR 2025) — diffusion-model interpretation via concept activation;
- **SAE-style explanations** (Cywiński et al., 2025; ref [11] in our paper) — sparse-autoencoder interpretability for diffusion models, run as a feature-attribution baseline.

We can also include the natural-language SAE captioning of [53] as a fifth baseline if space allows.

### W4. "Use more recent base models (e.g., Flux); SD 2.1 and ControlGAN are outdated."

We note that *FineXL is image-generator-agnostic* — we already evaluate diffusion (SD 2.1, SD 3.5 for f(C)), GAN (ControlGAN), and autoregressive (Anole-7B) families (Section G.3, Tables 8–9). The choice of SD 2.1 in the main tables was driven by NoiseCLR's available learned directions (Section 5), not by an inability of the method to handle newer models.

For camera-ready we will add Flux.1-dev as a fourth target architecture, with one of:
(a) NoiseCLR-style direction discovery on Flux's latent space (most consistent with Tables 1–4) — feasible because Flux exposes a clean rectified-flow latent;
(b) LoRA fine-tuning of Flux on three art-style datasets and reporting Tables 1–4 there.

We will lead with (b) since it also addresses R1-W1.

### W5. "The description of how ground-truth is obtained is not clear."

Ground truth is constructed in two scenarios:

- *Single-aspect (Section 6.1).* The "ground-truth level" of a personalized model is the scalar λ_e ∈ {-5, ..., 10} from Eq. (12). FineXL's predicted level for that concept, w_i, is compared against the rank of λ_e via mean-absolute-error on the rank ordering (Appendix F.3, Scenario 1). MAE on rank, not on raw value, is used precisely because no method should be expected to recover λ_e on the same numeric scale.
- *Multi-aspect (Section 6.2).* The "ground truth" is the mixture of style vectors {(λ_e1, λ_e2, ...)} used to generate the personalized images. Selection accuracy is measured against this coordinate (Figure 4 in appendix).

We will add a one-paragraph "Ground truth construction" box at the start of Section 5 in the camera-ready.

### W6 (minor). Same as W2 above.

Addressed in W2.

---

## Summary of camera-ready additions
| Concern | Addition |
|---|---|
| ptTg-W1, zLQn-W4 | Real DreamBooth/LoRA + Flux benchmark, ≥12 models |
| ptTg-W2, 6Ugt-W4 | Per-concept human study, 30 raters × 25 triples |
| ptTg-W3 | Higher-moment (covariance) decomposition discussion |
| ptTg-W4 | Quantitative SD-version comparison replacing Figure 12 |
| 6Ugt-W2 | Hallucination-injection ablation; explicit robustness paragraph |
| 6Ugt-W3 | Patch-level divergence pilot for localized/3D personalization |
| zLQn-W1 | Define w_i in text; orthogonality intuition box |
| zLQn-W2 | Reword "training-free" everywhere |
| zLQn-W3 | Add Llava-Diff, DiffEx, SAE-explanation baselines |
| zLQn-W5 | Explicit ground-truth construction paragraph |

None of these additions changes Algorithm 1 or any equation. We thank the reviewers and hope the clarifications and the new evaluations address the concerns sufficiently to lift the assessment.
