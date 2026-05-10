"""Scan runs/eval/* and produce a manifest.json for eval_pipeline.py."""
import json, os
from glob import glob

ROOT = "/home/haoming/finexl/runs/eval"

manifest = []
for d in sorted(glob(os.path.join(ROOT, "*"))):
    if not os.path.isdir(d):
        continue
    mid = os.path.basename(d)
    if mid == "base":
        continue
    # parse: <style>_step<level>  OR  <styleA>_<styleB>[..]_step<level>  for multi
    if "_step" not in mid:
        continue
    name, _, lvl = mid.rpartition("_step")
    level = int(lvl)
    parts = name.split("_")
    # Heuristic: allow multi-style names (e.g. "ukiyo_watercolor_step0200")
    style = parts[0] if len(parts) == 1 else "+".join(parts)
    mix = {p: 1.0 / len(parts) for p in parts} if len(parts) > 1 else {parts[0]: 1.0}
    manifest.append({
        "mid": mid,
        "eval_dir": d,
        "style": style if len(parts) == 1 else "_".join(parts),
        "level": level,
        "mix": mix,
    })

with open(os.path.join(ROOT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"manifest with {len(manifest)} models -> {ROOT}/manifest.json")
