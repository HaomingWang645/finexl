"""Smoke test: run FineXL on one (base, personalized) pair and print the
discovered concepts + weights. Uses Qwen2.5-VL as the proposer."""
import os, sys, json
sys.path.insert(0, "/home/haoming/finexl/code")

from glob import glob
from PIL import Image
from finexl import CLIPEncoder, divergence_vector_from_features, run_finexl
from finexl_full import QwenConceptProposer, PROBE_PROMPTS


def main():
    base_dir = sys.argv[1]
    pers_dir = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    base_paths = sorted(glob(os.path.join(base_dir, "*.png")))[:n]
    pers_paths = sorted(glob(os.path.join(pers_dir, "*.png")))[:n]
    base_imgs = [Image.open(p).convert("RGB") for p in base_paths]
    pers_imgs = [Image.open(p).convert("RGB") for p in pers_paths]
    print(f"using {len(base_imgs)} base + {len(pers_imgs)} personal images", flush=True)

    print("loading CLIP...", flush=True)
    clip_enc = CLIPEncoder("ViT-B-32", "openai")
    base_f = clip_enc.encode_images(base_imgs)
    pers_f = clip_enc.encode_images(pers_imgs)
    Vdiv = divergence_vector_from_features(base_f, pers_f)
    import numpy as np
    print(f"|Vdiv| = {np.linalg.norm(Vdiv):.4f}", flush=True)

    print("loading Qwen2.5-VL...", flush=True)
    qwen = QwenConceptProposer()
    print("proposing concepts...", flush=True)
    pairs = list(zip(pers_imgs, base_imgs))
    proposals = qwen.propose_for_pairs(pairs, max_pairs=4)
    print(f"VLM proposed: {proposals}", flush=True)

    print("running Algorithm 1...", flush=True)
    res = run_finexl(Vdiv, proposals, clip_enc, PROBE_PROMPTS,
                     e_ortho=0.30, e_decomp=0.20, verbose=True)
    print("\nKEPT concepts and weights:")
    for c, w in zip(res.concepts, res.weights):
        print(f"  {c:30s}  w={w:+.3f}")
    print(f"decomp error: {res.decomp_error:.3f}")


if __name__ == "__main__":
    main()
