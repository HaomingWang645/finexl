"""Generate eval images for multi-aspect LoRA models, append to manifest."""
import json, os
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


def attach_lora_fresh(unet, lora_pt_path):
    cfg = LoraConfig(r=16, lora_alpha=32,
                     target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                     lora_dropout=0.0, bias="none")
    unet = get_peft_model(unet, cfg)
    state = torch.load(lora_pt_path, map_location="cuda", weights_only=False)
    set_peft_model_state_dict(unet, state)
    return unet


def gen_for_model(pipe, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if len(glob(os.path.join(out_dir, "*.png"))) >= len(EVAL_PROMPTS):
        return
    for i, p in enumerate(EVAL_PROMPTS):
        out_path = os.path.join(out_dir, f"{i:02d}.png")
        if os.path.exists(out_path):
            continue
        g = torch.Generator("cuda").manual_seed(SEED_BASE + i)
        img = pipe(p, num_inference_steps=20, height=512, width=512, generator=g).images[0]
        img.save(out_path)


def main():
    multi_root = "/home/haoming/finexl/runs/lora_multi"
    manifest_extra = []
    for d in sorted(os.listdir(multi_root)):
        lp = os.path.join(multi_root, d, "step_0200", "lora.pt")
        if not os.path.exists(lp):
            continue
        styles = d.split("_")
        # Try to coalesce 'pixel_art' and 'oil_painting' which got split:
        coalesced = []
        i = 0
        while i < len(styles):
            if i + 1 < len(styles) and (styles[i], styles[i+1]) in [("pixel", "art"), ("oil", "painting")]:
                coalesced.append(styles[i] + "_" + styles[i+1])
                i += 2
            else:
                coalesced.append(styles[i])
                i += 1
        styles = coalesced
        n_aspects = len(styles)
        mid = "multi_" + d + "_step0200"
        out_dir = os.path.join("/home/haoming/finexl/runs/eval", mid)
        print(f"[{mid}] generating ...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir="/home/haoming/finexl/models/hf_cache",
            torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        pipe.unet = attach_lora_fresh(pipe.unet, lp)
        gen_for_model(pipe, out_dir)
        del pipe
        torch.cuda.empty_cache()
        manifest_extra.append({
            "mid": mid,
            "eval_dir": out_dir,
            "style": "+".join(styles),
            "level": 200,
            "mix": {s: 1.0 / n_aspects for s in styles},
            "n_aspects": n_aspects,
        })

    # Merge with existing manifest
    main_manifest = "/home/haoming/finexl/runs/eval/manifest.json"
    with open(main_manifest) as f:
        manifest = json.load(f)
    # Add n_aspects=1 to single-aspect entries
    for m in manifest:
        m.setdefault("n_aspects", 1)
    manifest.extend(manifest_extra)
    with open(main_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"appended {len(manifest_extra)} multi-aspect models; total {len(manifest)}")


if __name__ == "__main__":
    main()
