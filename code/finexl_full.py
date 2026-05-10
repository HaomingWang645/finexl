"""
Full FineXL pipeline as in Algorithm 1 of the paper, end-to-end:

  Input:  base-image folder, personal-image folder (paired by filename),
          path to a Qwen2.5-VL VLM checkpoint.
  Output: list of orthogonal concepts + their weights w_i, with the
          relative decomposition error.

This module is the "explanation" half. The image-generation half lives in
generate_eval.py / train_lora.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from glob import glob
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from finexl import (  # noqa: E402
    CLIPEncoder, divergence_vector_from_features, run_finexl,
)


# ----------------------------------------------------------------------------
# Qwen2.5-VL concept proposer (Section 4.2 of the paper, Figure 5 prompt)
# ----------------------------------------------------------------------------

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_TMPL = (
    "Compare the two images. Image 1 was produced by a personalized image "
    "generation model; Image 2 by the original base model. Both depict the "
    "same scene, but the personalized model has been fine-tuned with a "
    "specific artistic STYLE. Ignore content/scene differences and identify "
    "the artistic style of Image 1 (textures, brushwork, color palette, "
    "rendering style, art movement). Reply with up to 8 single-word "
    "adjectives or short phrases (each <=3 words) describing this style, "
    "comma-separated, no explanation."
)


class QwenConceptProposer:
    """Loads Qwen2.5-VL once; prompts it on (personal, base) pairs."""

    def __init__(self, device="cuda"):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor,
        )
        print(f"loading {QWEN_MODEL}...", flush=True)
        self.proc = AutoProcessor.from_pretrained(QWEN_MODEL)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map=device,
        ).eval()

    @torch.no_grad()
    def _raw_query(self, images: List[Image.Image], text_prompt: str,
                   max_new_tokens: int = 128) -> str:
        msgs = [
            {"role": "user", "content": (
                [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": text_prompt}]
            )},
        ]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=list(images),
                           padding=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        gen = out[:, inputs.input_ids.shape[1]:]
        return self.proc.batch_decode(gen, skip_special_tokens=True)[0]

    @torch.no_grad()
    def propose_for_pair(self, img_personal: Image.Image,
                         img_base: Image.Image) -> List[str]:
        s = self._raw_query([img_personal, img_base], PROMPT_TMPL,
                            max_new_tokens=64)
        # parse comma list
        items = [x.strip().strip(".") for x in re.split(r"[,;\n]+", s)]
        items = [x.lower() for x in items if x and len(x) <= 30]
        # keep only single words / short phrases (<=3 words)
        items = [x for x in items if len(x.split()) <= 3]
        return items

    @torch.no_grad()
    def propose_for_pairs(self, pairs: Sequence[Tuple[Image.Image, Image.Image]],
                          max_pairs: int = 8) -> List[str]:
        """Take union of concepts across multiple image pairs (Section 4.2)."""
        seen = set()
        result = []
        for ip, ib in pairs[:max_pairs]:
            for c in self.propose_for_pair(ip, ib):
                if c not in seen:
                    seen.add(c)
                    result.append(c)
        return result

    def free(self):
        del self.model, self.proc
        gc.collect()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Probe-prompt pool used for f(C) text-divergence (Eq. 6).
# Re-using the eval prompts isn't valid, so we use a separate probe set.
# ----------------------------------------------------------------------------

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
]


# ----------------------------------------------------------------------------
# CLI entry-point: run FineXL on one (base_dir, personal_dir) pair
# ----------------------------------------------------------------------------

def load_paired_images(base_dir: str, personal_dir: str
                       ) -> Tuple[List[Image.Image], List[Image.Image]]:
    bs = sorted(glob(os.path.join(base_dir, "*.png")))
    ps = sorted(glob(os.path.join(personal_dir, "*.png")))
    if not bs or not ps:
        raise FileNotFoundError(f"no images in {base_dir} or {personal_dir}")
    n = min(len(bs), len(ps))
    base_imgs = [Image.open(p).convert("RGB") for p in bs[:n]]
    pers_imgs = [Image.open(p).convert("RGB") for p in ps[:n]]
    return base_imgs, pers_imgs


def run_one(base_dir: str, personal_dir: str,
            clip_enc: CLIPEncoder, qwen: QwenConceptProposer | None,
            e_ortho: float = 0.30, e_decomp: float = 0.20,
            extra_candidates: List[str] | None = None,
            n_proposer_pairs: int = 6) -> dict:
    base_imgs, pers_imgs = load_paired_images(base_dir, personal_dir)
    n = len(base_imgs)

    # 1. V_div from CLIP image features
    base_f = clip_enc.encode_images(base_imgs)
    pers_f = clip_enc.encode_images(pers_imgs)
    Vdiv = divergence_vector_from_features(base_f, pers_f)

    # 2. Concept candidates (Section 4.2): VLM proposals + optional extras
    candidates = []
    if qwen is not None:
        pairs = list(zip(pers_imgs, base_imgs))
        proposals = qwen.propose_for_pairs(pairs, max_pairs=n_proposer_pairs)
        candidates.extend(proposals)
    if extra_candidates:
        for c in extra_candidates:
            if c not in candidates:
                candidates.append(c)
    if not candidates:
        raise RuntimeError("no concepts proposed")

    # 3. Algorithm 1
    res = run_finexl(
        Vdiv=Vdiv,
        candidate_concepts=candidates,
        encoder=clip_enc,
        probe_prompts=PROBE_PROMPTS,
        e_ortho=e_ortho,
        e_decomp=e_decomp,
        verbose=False,
    )

    return {
        "n_pairs": n,
        "Vdiv_norm": float(np.linalg.norm(Vdiv)),
        "candidates": candidates,
        "kept_concepts": res.concepts,
        "kept_weights": res.weights,
        "decomp_error": res.decomp_error,
        "rejected_for_orthogonality": [
            {"concept": c, "ortho_score": s} for c, s in res.rejected_concepts
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--personal_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--e_ortho", type=float, default=0.30)
    ap.add_argument("--e_decomp", type=float, default=0.20)
    ap.add_argument("--no_vlm", action="store_true",
                    help="skip Qwen-VL; use --extra_candidates as the entire pool")
    ap.add_argument("--extra_candidates", type=str, default=None,
                    help="comma-separated candidates added to the pool")
    args = ap.parse_args()

    print(f"loading CLIP...", flush=True)
    clip_enc = CLIPEncoder("ViT-B-32", "openai")

    qwen = None if args.no_vlm else QwenConceptProposer()

    extra = [c.strip() for c in args.extra_candidates.split(",")] \
        if args.extra_candidates else None

    res = run_one(args.base_dir, args.personal_dir, clip_enc, qwen,
                  e_ortho=args.e_ortho, e_decomp=args.e_decomp,
                  extra_candidates=extra)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("done. result at", args.out, flush=True)
    print("kept:", res["kept_concepts"])
    print("weights:", res["kept_weights"])
    print(f"decomp error: {res['decomp_error']:.3f}")


if __name__ == "__main__":
    main()
