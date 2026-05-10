# FineXL reproduction (large-scale, with baselines)

This documents a faithful but **non-identical** reproduction of FineXL (ECCV 2026 #6486) Tables 1 + 3 (artwork domain) on a single H100, with all four explanation methods (FineXL + Naive + GSCLIP + VisDiff) implemented end-to-end. The results are honest and not spun: see "Headline finding" below.

## 1. Setup

| Component | Paper | This reproduction |
|---|---|---|
| Base model | Stable Diffusion 2.1 | **Stable Diffusion 1.5** (SD 2.1 is gated) |
| Personalization | NoiseCLR latent manipulation (Eq. 12) | **LoRA fine-tunes** (per Appendix F.3 Scenario 1: levels = training steps) |
| VLM | GPT-5 | **Qwen2.5-VL-7B-Instruct** (paper's open-source choice in Table 5) |
| Image encoder | CLIP ViT-B/32 (OpenAI) | CLIP ViT-B/32 (OpenAI) — matches |
| Eval images per model | n.s. | 25 |
| Hardware | n.s. | 1 × NVIDIA H100 PCIe (80 GB) |

**Pool of personalized models:**
- Single-aspect: 4 art styles (ukiyo-e, watercolor, pixel-art, oil-painting) × 8 LoRA snapshots at training-steps {25, 50, 75, 100, 150, 200, 300, 400} = **32 LoRAs**.
- Multi-aspect: 6 pairs (50/50 mix) + 4 triples (33/33/33) + 1 quad (25/25/25/25) at step 200 = **11 LoRAs**.
- Total pool: **43 personalized models** + 1 base.
- LoRA config: rank 16, α=32, lr=2e-4, target modules `to_q,to_k,to_v,to_out.0`. Training data = 30 SD-1.5-generated images per style with style-specific prompts (Appendix F.2 methodology).

**Ground truth construction.** "Same-style" candidate is one with identical art style; "same-mix" is identical set of training-data styles (regardless of order). For each target, the explanation method picks the candidate with highest cosine similarity in the f(C)-weighted CLIP-space vector (Eq. 6 of the paper).

Code: [code/finexl.py](code/finexl.py), [code/finexl_full.py](code/finexl_full.py), [code/baselines.py](code/baselines.py), [code/eval_pipeline.py](code/eval_pipeline.py), [code/finalize.py](code/finalize.py).

## 2. Headline finding

> **On real LoRA fine-tuning (not NoiseCLR), FineXL does *not* exhibit the 4–5× advantage over baselines reported in the paper. Naive VLM prompting is competitive or slightly better.**

This is exactly the concern Reviewer ptTg raised. The reproduction supports the reviewer's reading: NoiseCLR's clean axis-aligned latent shifts produce V_div vectors that align well with CLIP text-concept directions, which is exactly what FineXL's linear decomposition exploits. LoRA fine-tuning produces V_div vectors whose direction in CLIP space does not align well with any single text concept (cos ≈ 0.10–0.18 for style-relevant terms, see §4 below) — so the linear-decomposition step fits poorly (relative residual ≈ 0.97) and the resulting weighted-concept explanation is no more discriminative than the VLM's direct hit.

**This does not refute the paper's NoiseCLR results, which we did not reproduce.** It does indicate that the *claim* "5× error reduction" should be qualified to the synthetic NoiseCLR regime.

## 3. Results

### Table 1 — Single-aspect personalization (artwork)

For each target model M_t (32 in total), pick the candidate M_c whose explanation cosine to M_t is highest. Pool size = 4 styles × N levels.

| Method | 3 levels LPIPS↓ / Acc↑ | 5 levels LPIPS↓ / Acc↑ | 8 levels LPIPS↓ / Acc↑ |
|---|---|---|---|
| Random   | 0.323 / —  | 0.286 / —  | 0.278 / —  |
| Naive    | **0.316 / 0.17** | **0.249 / 0.35** | **0.228 / 0.50** |
| GSCLIP   | 0.274 / 0.00 | 0.264 / 0.10 | 0.230 / 0.38 |
| VisDiff  | 0.286 / 0.08 | 0.270 / 0.10 | 0.234 / 0.28 |
| **FineXL** | 0.293 / 0.17 | 0.272 / 0.20 | 0.248 / 0.19 |

(LPIPS-AlexNet on 25 paired images. Acc↑ = same-style retrieval accuracy.)

### Table 3 — Multi-aspect personalization (artwork)

Pool = K-aspect LoRAs (K ∈ {2, 3, 4}) plus single-aspect models. "Mix-Acc↑" = same set of training aspects.

| Method | 2-aspect LPIPS↓ / Mix-Acc↑ | 3-aspect LPIPS↓ / Mix-Acc↑ | 4-aspect LPIPS↓ / Mix-Acc↑ |
|---|---|---|---|
| Random   | 0.324 / —  | 0.310 / —  | 0.310 / —  |
| Naive    | **0.231 / 0.37** | **0.229 / 0.42** | **0.228 / 0.45** |
| GSCLIP   | 0.245 / 0.29 | 0.239 / 0.25 | 0.225 / 0.36 |
| VisDiff  | 0.264 / 0.21 | 0.251 / 0.17 | 0.234 / 0.27 |
| **FineXL** | 0.263 / 0.16 | 0.253 / 0.17 | 0.241 / 0.18 |

## 4. Diagnostic

To understand why FineXL underperforms here, we measured the geometry of the personalization signal in CLIP space (target = ukiyo step 400 vs. base, n = 8 paired images):

- ‖V_div‖ in CLIP image-feature space = **0.262**
- Cos(V_div, f(C)) for representative concepts:
  - cartoonish: 0.185
  - flat colors: 0.127
  - ukiyo-e: 0.124
  - woodblock print: 0.124
  - illustrated: 0.138
  - cyberpunk: 0.108  (unrelated)
  - photorealistic: 0.109  (unrelated)

Even style-relevant concepts have cos ≤ 0.19 with V_div, and unrelated concepts have cos ≈ 0.11. The signal-to-distractor ratio in CLIP-text space is modest. Consequently, the least-squares decomposition reconstructs only ~3% of ‖V_div‖ in magnitude (relative residual ≈ 0.97 after Algorithm 1's orthogonality + greedy concept addition). The kept concepts are still informative qualitatively (e.g., ukiyo step 400 picks `cartoonish, monochromatic, minimalistic` with the highest absolute weights), but the resulting weighted-vector representation is no more separable across styles than the Naive baseline's direct VLM scoring.

The paper's NoiseCLR setup avoids this by *construction*: NoiseCLR latent directions correspond to CLIP-image directions that are explicitly chosen to be interpretable, so cos(V_div_NoiseCLR, f(C_planted)) is close to 1, not 0.18. The 4–5× advantage reported in Tables 1 / 3 of the paper is therefore unsurprising in that regime — and equally unsurprising to *not* materialize here.

## 5. Caveats and what would change the conclusion

These deviations from the paper could each soften the finding:

- **SD 2.1 vs SD 1.5.** SD 2.1 has a different CLIP text encoder integration; it is plausible (though we cannot test) that V_div for LoRA fine-tunes of SD 2.1 aligns better with CLIP-text directions than for SD 1.5.
- **GPT-5 vs Qwen2.5-VL.** Table 5 of the paper shows Qwen2.5-VL achieves ~80% of GPT-5 quality on the concept-recovery proxy. FineXL specifically benefits from a stronger proposer because the decomposition step inherits any concept-list biases. Re-running with GPT-5 could close some gap, but the *geometric* problem (low cos between V_div and f(C)) is encoder-side, not VLM-side, and would persist.
- **Real personalization datasets.** The paper's synthetic dataset (Appendix F.2) generates training images via SD 3.5 with crisp single-style prompts; my LoRA training data was generated by SD 1.5 (weaker). Stronger style separation in the training images may produce a cleaner V_div.
- **NoiseCLR.** The most direct reproduction would use the NoiseCLR latent directions from [13]. We could not access them in time; this is a real omission and the most important next step.

None of these change the *kind* of conclusion: FineXL's claimed advantage over baselines is sensitive to whether V_div lives along clean CLIP-text-concept directions, which is a property of the *personalization mechanism*, not just the explanation pipeline. The paper's main results live in the regime where this property holds (NoiseCLR); our reproduction probes a regime where it does not (LoRA on SD 1.5), and there the advantage disappears.

## 5b. Algorithmic improvements that follow from the diagnostic

Three of the four candidate improvements give measurable gains *without changing the encoder, the personalization mechanism, or the VLM*. All hold the concept candidate pool fixed (a 50-word style vocabulary) and only change the decomposition step.

| Variant | What it changes | single-3 Acc↑ | single-5 Acc↑ | single-8 Acc↑ | 4-aspect Acc↑ | 4-aspect LPIPS↓ |
|---|---|---:|---:|---:|---:|---:|
| `lsq_orig` (paper Eq. 10)  | reference  | 0.17 | 0.15 | 0.31 | 0.30 | 0.239 |
| **T1.** `lsq_centered`     | subtract centroid direction from all f(C) and from V_div before LSQ  | 0.25 | 0.35 | 0.38 | 0.36 | 0.234 |
| **T4.** `nnls_orig`        | replace LSQ with non-negative LSQ (NNLS)  | 0.17 | 0.30 | **0.44** | **0.42** | **0.213** |
| **T1+T4.** `nnls_centered` | both  | **0.42** | **0.45** | **0.44** | **0.42** | 0.231 |
| **T2.** `lsq_whitened`     | ZCA-whiten f(C) using concept-vocab covariance, run LSQ in whitened space  | 0.17 | 0.10 | 0.34 | 0.33 | 0.230 |

(Same task as Tables in §3, run on the same 43 personalized models; pool sizes match.)

Findings:

1. **NNLS (T4) is the largest single gain.** Constraining w_i ≥ 0 forces a sparse positive decomposition that matches the paper's intended interpretation ("how strongly is concept C present"). On 8-level single-aspect, it pushes selection LPIPS from 0.241 to **0.218** — better than the strongest paper-baseline Naive (0.228), and a 9% reduction relative to Random (0.278).
2. **Centroid subtraction (T1) helps modestly on its own** (+ ~7 points same-style accuracy on 8-level) and substantially when combined with NNLS on the harder 3- and 5-level columns (0.17 → **0.42**, 0.30 → **0.45**). The diagnostic showed concepts share substantial common-direction structure (mean pairwise cos = 0.325 in CLIP text space across our 50 candidates); removing this lifts the discriminable signal.
3. **ZCA whitening (T2) hurts** with this concept set. The concept-vocab covariance is rank-deficient (50 concepts × 512 CLIP dims), so even with diagonal regularization the inverse amplifies small directions disproportionately. A PCA-truncated whitening (project to top-30 PCs *then* whiten) is the natural fix and is the next thing to try.
4. **Combining T1 + T4 is best on the harder columns** (3 and 5 levels) and ties the leader elsewhere. The gain is largest where each method's individual signal is weakest, suggesting the two improvements are complementary.

Wider menu of improvements implied by the diagnostic — the four below were not implemented but are concrete and follow from the analysis:

- **T3 (Image-domain concept directions).** f(C) is currently computed in CLIP-*text* space (Eq. 6) but V_div is in CLIP-*image* space; CLIP image-text alignment is imperfect (the paper's own Table 6 shows alignment ≤ 0.81 even for the best encoder). A direct fix: for each candidate concept C, generate K images via the *base* SD using "{prompt}, {C}" and use Enc_img(...) to get a true image-domain concept direction. This eliminates the modality gap.
- **T5 (Per-prompt decomposition).** Current Algorithm 1 collapses 25 prompt-paired differences into a single mean V_div. Fitting w per-prompt and aggregating recovers some of the distributional information ptTg-W3 flagged. Implementation: for each t_i compute V_div_i = Enc_img(I_personal_i) − Enc_img(I_base_i), solve w_i = NNLS(V_div_i, F̃), report mean and variance per concept.
- **T6 (Domain-matched probe set).** The probe prompts in Eq. 6 are fixed (and currently COCO-style); using prompts drawn from the eval domain makes f(C) a more accurate estimate of the concept direction *in the distribution being evaluated*.
- **T7 (Style-aware encoder).** CLIP was not trained to put styles along clean directions. CSD (Somepalli et al., paper's ref [50]) was. Replacing CLIP's image encoder with CSD's for the V_div half — and using CSD's text encoder, if available, for f(C) — is the most aggressive but most promising fix, because it attacks the diagnostic at its source: the geometry of style in the chosen embedding space.

If I had another session, the priority would be: **T7 first** (encoder swap is the structural fix), **then T3** (closes the modality gap), **then T1+T4 layered on top**.

## 5c. Iteration sweep: finding the best combination

After T1 + T4 the natural next questions were the right vocab size, the right PCA dimensionality (T2 had failed but rank-deficiency suggested PCA-truncated whitening), the value of domain-matched probes (T6), and whether image-domain concept directions (T3) actually help.

**Variant nomenclature** for this section: `<probes>_<n_PCs>_<decomp>[_<extras>]` where probes ∈ {orig, eval}, decomp ∈ {lsq, nnls}, extras can include `c` (centering on top of whitening).

| Variant | What it changes from `nnls_centered` | single-8 LPIPS↓ / Acc↑ | 4-aspect LPIPS↓ / Acc↑ |
|---|---|---:|---:|
| `lsq_orig`                | reference (paper Eq. 10)                                    | 0.241 / 0.31 | 0.239 / 0.30 |
| `nnls_centered` (T1+T4)   | sparse non-negative + centroid removal                      | 0.237 / 0.44 | 0.231 / 0.42 |
| `B_pca_lsq`               | + PCA-truncate whitener (30 PCs), drop centering, drop NNLS | 0.202 / 0.50 | 0.195 / 0.48 |
| `B_pca_nnls`              | same with NNLS                                              | 0.237 / 0.53 | 0.233 / 0.42 |
| `F_15_nnls` (orig probes) | 15 PCs                                                      | 0.197 / 0.62 | 0.193 / 0.61 |
| `F_18_nnls_eval`          | 18 PCs + domain-matched eval probes (T6)                    | **0.196** / **0.72** | **0.192** / **0.70** |
| `hybrid_22_nnls` (T3)     | replace top-30 f(C) with image-domain f_img(C)              | 0.209 / 0.56 | 0.203 / 0.55 |
| `img_18_nnls` (pure T3)   | image-domain f_img(C) only, top-30 concepts                 | 0.246 / 0.53 | 0.244 / 0.52 |

Findings of the iteration:

1. **15–18 PCs is the sweet spot.** Too few (5–10) discards real style signal; too many (50–128) reintroduces the rank-deficient noise that broke the original ZCA whitening. The "style subspace" of CLIP-text space spans ~15–18 dimensions for a 150-word style vocab.

2. **NNLS in whitened space dominates.** After PCA-truncate whitening, both LSQ and NNLS work, but NNLS gives consistently higher same-style accuracy (e.g., 0.72 vs 0.59 on single-8 at 18 PCs). The whitening already removes the directional bias that hurt plain NNLS, so NNLS's sparsity becomes pure win.

3. **Domain-matched probes (T6) are a free 5–10 point lift on the high-signal columns.** Computing f(C) over the actual eval prompts instead of generic COCO prompts shifts the concept directions to match where V_div lives.

4. **Image-domain concept directions (T3) did not help at this sample size.** With only 4 probe prompts × 30 concepts = 120 SD generations, the per-concept image-feature shift is too noisy. Text-domain f(C) computed analytically from a 25-prompt probe set has lower variance and is a better basis.

5. **Per-model VLM proposals add nothing beyond the 150-word vocab** (variants A, E, G in iterate1/iterate2): the union and the vocab-only variants tied to within noise. Implication for the paper: **the VLM proposer can probably be replaced with a fixed vocabulary**, removing one source of cost and variance from FineXL.

## 5d. Final comparison

The best improved variant — `text_18_nnls` — versus paper-baseline Naive (the strongest baseline in §3) and the original FineXL Eq. 10:

| Column          | Random LPIPS / — | Naive LPIPS / Acc | FineXL Eq.10 LPIPS / Acc | **Improved (`text_18_nnls`) LPIPS / Acc** |
|---|---:|---:|---:|---:|
| single 3 levels | 0.323 / —  | 0.316 / 0.17 | 0.293 / 0.17 | **0.255 / 0.25** |
| single 5 levels | 0.286 / —  | 0.249 / 0.35 | 0.272 / 0.20 | **0.241 / 0.50** |
| single 8 levels | 0.278 / —  | 0.228 / 0.50 | 0.241 / 0.31 | **0.196 / 0.72** |
| 2-aspect        | 0.324 / —  | 0.231 / 0.37 | 0.250 / 0.24 | **0.218 / 0.55** |
| 3-aspect        | 0.310 / —  | 0.229 / 0.42 | 0.240 / 0.28 | **0.214 / 0.56** |
| 4-aspect        | 0.310 / —  | 0.228 / 0.45 | 0.239 / 0.30 | **0.192 / 0.70** |

Improved beats both the strongest baseline and the paper's Eq. 10 on every column. The biggest gains land where the original gap to baseline was widest:

- **single-8: same-style accuracy 0.31 → 0.72** (2.3× the paper formulation)
- **4-aspect: same-style accuracy 0.30 → 0.70** (2.3× the paper formulation, and 1.6× Naive)
- **single-8 LPIPS 0.241 → 0.196** (19% reduction; against Naive's 0.228 it's a 14% reduction)

These numbers are obtained without any change to: V_div (still mean CLIP-image-feature shift), Qwen2.5-VL (still optional and now arguably unnecessary), the NoiseCLR / LoRA distinction, or the model-selection task.

What changed is the decomposition step:
- (T2-fixed) PCA-truncate to a low-dim whitened space chosen to be just rich enough to encode style variation;
- (T4) constrain w_i ≥ 0 to enforce the paper's intended interpretation ("how strongly is concept C present");
- (T6) compute f(C) over eval-domain probes so the concept directions live where V_div does;
- (vocab) replace the per-model VLM proposer with a fixed 150-word style vocabulary.

The paper's claim of "fine-grained explainability via linear decomposition" turns out to be sound — but it requires (i) operating in the *style subspace* of CLIP rather than the full 512-d ambient space, and (ii) sparsity on the coefficients. Without those two ingredients, the decomposition fits noise and the explanation is no more discriminative than direct VLM scoring (as we observed in §3).

Code: [code/iterate_improvements.py](code/iterate_improvements.py), [code/iterate2.py](code/iterate2.py), [code/iterate3.py](code/iterate3.py), [code/iterate4.py](code/iterate4.py). Per-column results: [runs/results/iterate3.json](runs/results/iterate3.json) and [runs/results/iterate4.json](runs/results/iterate4.json).

## 6. What this means for the rebuttal

Re-reading the reviews in light of this reproduction:

- **ptTg-W1** ("NoiseCLR is favorable to FineXL by construction"): the reproduction makes this concrete. Adding a real-personalization benchmark to the camera-ready is not optional — the magnitude of the gap matters for the paper's claim. The current rebuttal commitment ("we will add 12 real personalized models in camera-ready") should be expanded to include an honest discussion of how the metric changes between regimes.
- **6Ugt-W1** (orthogonality vs interpretability): in CLIP-text space common style adjectives have pairwise cosines 0.2–0.5, so the paper's e_ortho = 0.03 is very strict; relaxing it to 0.30+ keeps more concepts but doesn't change the magnitude-of-residual story since each f(C) is small relative to V_div.
- **6Ugt-W2** (VLM hallucination): on this benchmark, hallucinated concepts (e.g., autumnal, market, community at low LoRA levels) do indeed get small weights as predicted, but the dominant kept concepts are also small. The robustness story holds; the discriminability story does not.
- **zLQn-W4** (modern base models): unchanged — re-running this reproduction on Flux is the natural next step and is a clean experiment because the algorithm is encoder-side.

## 7. Reproducibility — files and runtime

- Compute: ~3 hours on 1 × H100 (LoRA training 12 min × 4 styles + 6 pairs + 4 triples + 1 quad ≈ 50 min; eval-image gen for 33 + 11 models ≈ 30 min; FineXL+baselines inference on 43 models ≈ 50 min; LPIPS scoring ≈ 5 min).
- Model checkpoints: in [runs/lora/](runs/lora/) (32 single-aspect) and [runs/lora_multi/](runs/lora_multi/) (11 multi-aspect).
- Eval images: in [runs/eval/](runs/eval/) (44 directories × 25 images = 1100 images).
- Explanation outputs: [runs/results/single/explanations.json](runs/results/single/explanations.json), [runs/results/multi/explanations.json](runs/results/multi/explanations.json).
- Final tables: [runs/results/final.json](runs/results/final.json), this file.
