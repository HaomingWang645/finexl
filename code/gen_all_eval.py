"""
Generate evaluation images for the BASE model and every (style, level)
LoRA-personalized model. Uses fixed prompts + fixed seeds so target/candidate
images are pairwise comparable across models.

Reuses the SD pipeline once, only swapping LoRA adapters per model. Avoids
reloading the full SD weights for every personalized model.
"""
import json
import os
from glob import glob

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image


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
SEED_BASE = 42
INFER_STEPS = 20

STYLES = ["ukiyo", "watercolor", "pixel_art", "oil_painting"]
LEVELS = [25, 50, 75, 100, 150, 200, 300, 400]


def build_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="/home/haoming/finexl/models/hf_cache",
        torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def attach_lora_fresh(unet, lora_pt_path):
    cfg = LoraConfig(r=16, lora_alpha=16,
                     target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                     lora_dropout=0.0, bias="none")
    unet = get_peft_model(unet, cfg)
    state = torch.load(lora_pt_path, map_location="cuda", weights_only=False)
    set_peft_model_state_dict(unet, state)
    return unet


def gen_for_model(pipe, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    done = sorted(glob(os.path.join(out_dir, "*.png")))
    if len(done) >= len(EVAL_PROMPTS):
        return
    for i, p in enumerate(EVAL_PROMPTS):
        out_path = os.path.join(out_dir, f"{i:02d}.png")
        if os.path.exists(out_path):
            continue
        g = torch.Generator("cuda").manual_seed(SEED_BASE + i)
        img = pipe(p, num_inference_steps=INFER_STEPS,
                   height=512, width=512, generator=g).images[0]
        img.save(out_path)


def main():
    pipe = build_pipe()

    base_eval = "/home/haoming/finexl/runs/eval/base"
    print(f"generating BASE eval -> {base_eval}", flush=True)
    gen_for_model(pipe, base_eval)

    manifest = []
    pristine_unet = pipe.unet  # save the original

    for style in STYLES:
        for level in LEVELS:
            lora_path = f"/home/haoming/finexl/runs/lora/{style}/step_{level:04d}/lora.pt"
            if not os.path.exists(lora_path):
                print(f"  missing {lora_path}, skip", flush=True)
                continue
            mid = f"{style}_step{level:04d}"
            out_dir = f"/home/haoming/finexl/runs/eval/{mid}"
            print(f"[{mid}] generating ...", flush=True)
            # rebuild pipe.unet by loading SD again -- much cleaner than detach
            pipe2 = StableDiffusionPipeline.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                cache_dir="/home/haoming/finexl/models/hf_cache",
                torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False,
            ).to("cuda")
            pipe2.set_progress_bar_config(disable=True)
            pipe2.unet = attach_lora_fresh(pipe2.unet, lora_path)
            gen_for_model(pipe2, out_dir)
            del pipe2
            torch.cuda.empty_cache()
            manifest.append({
                "mid": mid, "eval_dir": out_dir,
                "style": style, "level": level,
                "mix": {style: 1.0},
            })

    out = "/home/haoming/finexl/runs/eval/manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {out}; {len(manifest)} models", flush=True)


if __name__ == "__main__":
    main()
