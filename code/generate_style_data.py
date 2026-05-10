"""
Generate per-style training images using base SD 1.5 with style prompts.
Mirrors Appendix F.2 of the paper: "object description + image style description".
"""
import os
import json
import argparse
import torch
from diffusers import StableDiffusionPipeline


# 4 distinct art styles for the artwork domain (from paper Fig. 2 in Appendix)
STYLES = {
    "ukiyo": {
        "prompt_suffix": ", in the style of a traditional Japanese ukiyo-e woodblock print, flat colors, bold black outlines, decorative patterns",
        "trigger": "ukiyo style",
    },
    "watercolor": {
        "prompt_suffix": ", as a soft watercolor painting, washed pastel tones, visible brush strokes, paper texture, dreamy",
        "trigger": "watercolor style",
    },
    "pixel_art": {
        "prompt_suffix": ", as 16-bit pixel art, sharp pixelated edges, limited palette, retro video game aesthetic",
        "trigger": "pixel art style",
    },
    "oil_painting": {
        "prompt_suffix": ", as a classical oil painting, thick impasto brushwork, rich saturated colors, chiaroscuro lighting, Rembrandt-like",
        "trigger": "oil painting style",
    },
}

# 30 short COCO-style object/scene prompts
OBJECTS = [
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
    "a panda eating bamboo in a forest",
    "a steaming bowl of noodles on a table",
    "an astronaut floating above the earth",
    "a vintage red bicycle leaning on a tree",
    "a wolf howling on a snowy hill",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/haoming/finexl/data/styles")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()

    print("loading SD 1.5 ...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="/home/haoming/finexl/models/hf_cache",
        torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for style_id, style in STYLES.items():
        sdir = os.path.join(args.out, style_id)
        os.makedirs(sdir, exist_ok=True)
        print(f"\n[{style_id}] generating {len(OBJECTS)} images ...", flush=True)
        for i, obj in enumerate(OBJECTS):
            prompt = obj + style["prompt_suffix"]
            out_path = os.path.join(sdir, f"{i:02d}.png")
            if os.path.exists(out_path):
                continue
            g = torch.Generator("cuda").manual_seed(1000 + i)
            img = pipe(prompt, num_inference_steps=args.steps,
                       height=args.height, width=args.width, generator=g).images[0]
            img.save(out_path)
            manifest.append({"style": style_id, "path": out_path,
                             "prompt": prompt, "object": obj})
            if (i+1) % 5 == 0:
                print(f"  [{style_id}] {i+1}/{len(OBJECTS)}", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"styles": STYLES, "objects": OBJECTS, "manifest": manifest}, f, indent=2)
    print("\ndone. manifest at", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
