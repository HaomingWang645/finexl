"""
Generate evaluation images from the BASE model and from each LoRA-snapshot
personalized model, using a fixed shared eval prompt set + fixed seeds so
images from base and personalized are pairwise comparable.

Saves into runs/eval/<model_id>/{0..N-1}.png
"""
import argparse
import json
import os
import sys
from glob import glob

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

sys.path.insert(0, os.path.dirname(__file__))


# 25 fixed evaluation prompts (different from training; no style suffix).
EVAL_PROMPTS = [
    "a cyclist racing along a coastal road",
    "an elderly woman knitting by a window",
    "a child building a sandcastle on the beach",
    "a busy farmer's market in autumn",
    "a vintage tractor in a wheat field",
    "a cat curled up on a stack of books",
    "two friends laughing at a dinner table",
    "a violinist on a foggy bridge",
    "a hiker watching sunrise from a cliff",
    "a small bakery with fresh bread on display",
    "a fox stepping carefully across snow",
    "a teenager skateboarding under streetlights",
    "a fisherman pulling nets at dusk",
    "an astronaut planting a flag on the moon",
    "a teacher writing equations on a chalkboard",
    "a violinist playing in a quiet park",
    "a chef tossing pasta in a pan",
    "a ballerina spinning on a wooden floor",
    "a deer drinking from a stream",
    "a young girl flying a paper airplane",
    "a fountain in a city plaza at night",
    "a basketball player jumping for a dunk",
    "a samurai standing in a bamboo forest",
    "a couple dancing on a rooftop at sunset",
    "an old typewriter on a cluttered desk",
]


def load_lora(unet, lora_pt_path):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    unet = get_peft_model(unet, cfg)
    state = torch.load(lora_pt_path, map_location="cuda", weights_only=False)
    set_peft_model_state_dict(unet, state)
    return unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=str, default=None,
                    help="path to lora.pt; omit for base model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--n_prompts", type=int, default=25)
    ap.add_argument("--seed_base", type=int, default=42)
    args = ap.parse_args()

    print(f"loading SD 1.5 ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="/home/haoming/finexl/models/hf_cache",
        torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    if args.lora:
        print(f"loading LoRA from {args.lora}", flush=True)
        pipe.unet = load_lora(pipe.unet, args.lora)

    os.makedirs(args.out, exist_ok=True)
    prompts = EVAL_PROMPTS[: args.n_prompts]

    info = []
    for i, p in enumerate(prompts):
        out_path = os.path.join(args.out, f"{i:02d}.png")
        if os.path.exists(out_path):
            continue
        g = torch.Generator("cuda").manual_seed(args.seed_base + i)
        img = pipe(p, num_inference_steps=args.steps, height=512, width=512,
                   generator=g).images[0]
        img.save(out_path)
        info.append({"idx": i, "prompt": p, "path": out_path})

    with open(os.path.join(args.out, "_info.json"), "w") as f:
        json.dump({"prompts": prompts, "info": info,
                   "lora": args.lora, "seed_base": args.seed_base}, f, indent=2)
    print(f"done. {len(prompts)} images at {args.out}", flush=True)


if __name__ == "__main__":
    main()
