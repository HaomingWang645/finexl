"""
LoRA fine-tune SD 1.5 UNet on a per-style folder, saving snapshots at multiple
training-step counts. Each snapshot becomes a "personalization level" per
Appendix F.3 Scenario 1: "personalized levels relate to the number of training
steps before overfitting".

Trains a single LoRA on a single dataset, but saves separate adapters at each
target step count (e.g., {25, 50, 75, 100, 150, 200, 300, 400}). This is much
faster than training 8 independent runs since the model is the same up to step
counter.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import (
    StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer


class StyleDataset(Dataset):
    def __init__(self, paths, tokenizer, caption: str, size=512):
        self.paths = paths
        self.tokenizer = tokenizer
        self.caption = caption
        self.tx = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x = self.tx(img)
        ids = self.tokenizer(
            self.caption, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        return x, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshots", type=int, nargs="+",
                    default=[25, 50, 75, 100, 150, 200, 300, 400])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caption", default="a photo, art")
    ap.add_argument("--mixed_dirs", type=str, default=None,
                    help="comma-separated list of style dirs for multi-aspect; if set, mixes them")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.float16

    cache = "/home/haoming/finexl/models/hf_cache"
    base = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"loading SD 1.5 components from {base}...", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer", cache_dir=cache)
    text_enc = CLIPTextModel.from_pretrained(base, subfolder="text_encoder",
                                             torch_dtype=dtype, cache_dir=cache).to(device)
    text_enc.requires_grad_(False).eval()
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae",
                                        torch_dtype=dtype, cache_dir=cache).to(device)
    vae.requires_grad_(False).eval()
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet",
                                                torch_dtype=torch.float32, cache_dir=cache).to(device)
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler", cache_dir=cache)

    # LoRA on UNet attention layers
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # Dataset
    if args.mixed_dirs:
        paths = []
        for d in args.mixed_dirs.split(","):
            paths.extend(sorted(glob(os.path.join(d, "*.png"))))
    else:
        paths = sorted(glob(os.path.join(args.style_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no images found in {args.style_dir}/{args.mixed_dirs}")
    print(f"training on {len(paths)} images", flush=True)
    ds = StyleDataset(paths, tokenizer, args.caption)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)

    opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    # We may need many epochs to accumulate enough optimizer steps for large
    # snapshots like 400.
    target_steps = max(args.snapshots)
    n_per_epoch = max(1, len(ds) // args.bs)
    n_epochs = math.ceil(target_steps / n_per_epoch)
    print(f"will train ~{n_epochs} epochs to reach {target_steps} steps", flush=True)

    os.makedirs(args.out, exist_ok=True)

    snap_set = set(args.snapshots)
    step = 0
    saved = []
    unet.train()
    for epoch in range(n_epochs):
        for x, ids in dl:
            x = x.to(device, dtype=dtype)
            ids = ids.to(device)

            with torch.no_grad():
                latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                enc_h = text_enc(ids)[0]

            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps,
                              (latents.shape[0],), device=device).long()
            noisy = noise_sched.add_noise(latents, noise, t)

            with torch.amp.autocast("cuda", dtype=dtype):
                model_out = unet(noisy.float(), t, encoder_hidden_states=enc_h.float()).sample
                target = noise.float()
                loss = F.mse_loss(model_out, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
            opt.step()

            step += 1
            if step in snap_set:
                snap_dir = os.path.join(args.out, f"step_{step:04d}")
                os.makedirs(snap_dir, exist_ok=True)
                lora_state = get_peft_model_state_dict(unet)
                torch.save(lora_state, os.path.join(snap_dir, "lora.pt"))
                # also save args for traceability
                with open(os.path.join(snap_dir, "info.json"), "w") as f:
                    json.dump({"step": step, "loss": float(loss.item()),
                               "n_train_imgs": len(paths),
                               "rank": args.rank, "lr": args.lr,
                               "caption": args.caption,
                               "style_dir": args.style_dir or args.mixed_dirs}, f, indent=2)
                saved.append(step)
                print(f"  saved step {step}, loss {loss.item():.4f}", flush=True)
            if step >= target_steps:
                break
            if step % 25 == 0:
                print(f"  step {step}/{target_steps}, loss {loss.item():.4f}", flush=True)
        if step >= target_steps:
            break

    print(f"done. saved snapshots at steps: {saved}")


if __name__ == "__main__":
    main()
