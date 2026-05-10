"""Generate one image per LoRA snapshot to visually inspect the level progression."""
import os, sys, torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image

style = sys.argv[1] if len(sys.argv) > 1 else "ukiyo"
src = sys.argv[2] if len(sys.argv) > 2 else "/home/haoming/finexl/runs/lora_test"
prompt = "a cyclist racing along a coastal road"

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    cache_dir="/home/haoming/finexl/models/hf_cache",
    torch_dtype=torch.float16,
    safety_checker=None, requires_safety_checker=False,
).to("cuda")
pipe.set_progress_bar_config(disable=True)

# base
g = torch.Generator("cuda").manual_seed(42)
img_base = pipe(prompt, num_inference_steps=20, generator=g).images[0]
imgs = [img_base.resize((256, 256))]
labels = ["base"]

snaps = sorted(os.listdir(os.path.join(src, style)))
for snap in snaps:
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=0.0, bias="none")
    pipe2 = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        cache_dir="/home/haoming/finexl/models/hf_cache",
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to("cuda")
    pipe2.set_progress_bar_config(disable=True)
    pipe2.unet = get_peft_model(pipe2.unet, cfg)
    state = torch.load(os.path.join(src, style, snap, "lora.pt"), map_location="cuda", weights_only=False)
    set_peft_model_state_dict(pipe2.unet, state)
    g = torch.Generator("cuda").manual_seed(42)
    img = pipe2(prompt, num_inference_steps=20, generator=g).images[0]
    imgs.append(img.resize((256, 256)))
    labels.append(snap)
    del pipe2
    torch.cuda.empty_cache()

# tile
W = 256 * len(imgs)
strip = Image.new("RGB", (W, 256))
for i, im in enumerate(imgs):
    strip.paste(im, (256 * i, 0))
out = f"/home/haoming/finexl/results/visual_check_{style}.png"
strip.save(out)
print("saved", out, "labels:", labels)
