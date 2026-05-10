"""
Final aggregation:
  - Loads explanations.json (FineXL + Naive + GSCLIP + VisDiff vectors per model)
  - For each "table column" (level/aspect subset), runs the model-selection task
  - Computes LPIPS to selected candidate (lower is better) and exact-match accuracy
  - Writes REPRODUCTION.md with tables.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def load_jsons():
    out = {}
    out["manifest"] = json.load(open("/home/haoming/finexl/runs/eval/manifest.json"))
    expl_paths = [
        "/home/haoming/finexl/runs/results/single/explanations.json",
        "/home/haoming/finexl/runs/results/multi/explanations.json",
    ]
    expl = {}
    for p in expl_paths:
        if os.path.exists(p):
            expl.update(json.load(open(p)))
    out["explanations"] = expl
    return out


def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def selection_task(models_subset, explanations, lpips_fn=None, method="FineXL"):
    """For each target, pick the candidate with the highest cosine similarity
    between explanation vectors. Return mean LPIPS (if scorer provided),
    same-style match accuracy, and same-mix accuracy."""
    rows = []
    same_style_match = 0
    same_mix_match = 0
    for tgt in models_subset:
        tv = explanations[tgt["mid"]][method]["vec"]
        best_sim = -2; best = None
        for cand in models_subset:
            if cand["mid"] == tgt["mid"]:
                continue
            cv = explanations[cand["mid"]][method]["vec"]
            s = cosine(tv, cv)
            if s > best_sim:
                best, best_sim = cand, s
        # Same-style: target style equals candidate style
        ss = (tgt["style"] == best["style"])
        # Same-mix: same set of aspects (regardless of order)
        sm = (set(tgt["mix"].keys()) == set(best["mix"].keys()))
        same_style_match += int(ss)
        same_mix_match += int(sm)
        rows.append({
            "target": tgt["mid"], "selected": best["mid"],
            "sim": best_sim, "same_style": ss, "same_mix": sm,
            "tgt_style": tgt["style"], "tgt_level": tgt["level"],
            "sel_style": best["style"], "sel_level": best["level"],
        })
    n = len(models_subset)
    return {
        "n": n,
        "same_style_acc": same_style_match / n,
        "same_mix_acc": same_mix_match / n,
        "rows": rows,
    }


def lpips_scorer():
    import torch
    from torchvision import transforms as T
    from PIL import Image
    from glob import glob
    import lpips as lpips_pkg
    net = lpips_pkg.LPIPS(net="alex").to("cuda").eval()
    tx = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    cache = {}
    def load(d):
        if d not in cache:
            paths = sorted(glob(os.path.join(d, "*.png")))
            ts = []
            for p in paths:
                ts.append(tx(Image.open(p).convert("RGB")).unsqueeze(0).to("cuda")*2-1)
            cache[d] = ts
        return cache[d]
    @torch.no_grad()
    def score(dir_a, dir_b):
        a = load(dir_a); b = load(dir_b)
        n = min(len(a), len(b))
        return float(np.mean([net(a[i], b[i]).mean().item() for i in range(n)]))
    return score


def main():
    data = load_jsons()
    M = data["manifest"]
    E = data["explanations"]
    methods = ["Random", "Naive", "GSCLIP", "VisDiff", "FineXL"]
    print(f"loaded {len(M)} models, {len(E)} explanations", flush=True)

    # Filter to models that have explanations
    M = [m for m in M if m["mid"] in E]
    print(f"intersected: {len(M)} models with explanations", flush=True)

    # Define column subsets
    columns = []
    # Table 1 columns (single-aspect only)
    M_single = [m for m in M if m.get("n_aspects", 1) == 1]
    for K, levels in [(3, [25, 100, 400]),
                      (5, [25, 75, 100, 200, 400]),
                      (8, [25, 50, 75, 100, 150, 200, 300, 400])]:
        sub = [m for m in M_single if m["level"] in levels]
        columns.append((f"single-aspect, {K} levels", sub))

    # Table 3 columns (multi-aspect)
    for k in [2, 3, 4]:
        sub = [m for m in M if m.get("n_aspects", 1) == k]
        # combine with single-aspect for richer pool (paper does this)
        sub = sub + M_single
        columns.append((f"{k}-aspect (+ single-aspect pool)", sub))

    print("loading LPIPS scorer...", flush=True)
    lpips_fn = lpips_scorer()

    rng = np.random.default_rng(0)

    # Random baseline: average pairwise LPIPS over random pairs in each column
    results = {}
    for label, sub in columns:
        results[label] = {}
        for m in methods[1:]:  # skip Random
            r = selection_task(sub, E, method=m)
            # LPIPS to selected
            lps = []
            for row in r["rows"]:
                ta = next(x for x in sub if x["mid"] == row["target"])["eval_dir"]
                ca = next(x for x in sub if x["mid"] == row["selected"])["eval_dir"]
                lps.append(lpips_fn(ta, ca))
            r["lpips"] = float(np.mean(lps))
            results[label][m] = r
            print(f"  [{label}] {m:8s}  same-style={r['same_style_acc']:.2f}  "
                  f"same-mix={r['same_mix_acc']:.2f}  lpips={r['lpips']:.3f}",
                  flush=True)

        # Random
        rnd_lps = []
        for tgt in sub:
            others = [m for m in sub if m["mid"] != tgt["mid"]]
            cand = others[int(rng.integers(0, len(others)))]
            rnd_lps.append(lpips_fn(tgt["eval_dir"], cand["eval_dir"]))
        results[label]["Random"] = {"lpips": float(np.mean(rnd_lps))}
        print(f"  [{label}] Random   lpips={results[label]['Random']['lpips']:.3f}",
              flush=True)

    # Save full
    with open("/home/haoming/finexl/runs/results/final.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown tables
    md = ["# FineXL reproduction results\n"]
    md.append("## Table 1: Single-aspect personalization, artwork (LPIPS↓ on selection task)\n")
    md.append("Pool = 4 styles × N levels. For each target model, FineXL/baseline picks the closest candidate by explanation cosine; we report (a) LPIPS between target and selected images and (b) accuracy of selecting the same-style candidate.\n")

    cols_t1 = [c for c in columns if c[0].startswith("single-aspect")]
    md.append("| Method | " + " | ".join(f"{c[0].split(', ')[1]} LPIPS↓ / Acc↑"
                                          for c in cols_t1) + " |")
    md.append("| --- " + "| --- " * len(cols_t1) + "|")
    for m in methods:
        cells = [m]
        for label, _ in cols_t1:
            r = results[label].get(m, {})
            if m == "Random":
                cells.append(f"{r.get('lpips', float('nan')):.3f} / -")
            else:
                cells.append(f"{r.get('lpips', float('nan')):.3f} / "
                             f"{r.get('same_style_acc', float('nan')):.2f}")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n## Table 3: Multi-aspect personalization, artwork (LPIPS↓ on selection task)\n")
    cols_t3 = [c for c in columns if "aspect (" in c[0]]
    md.append("| Method | " + " | ".join(f"{c[0].split(' (')[0]} LPIPS↓ / Mix-Acc↑"
                                          for c in cols_t3) + " |")
    md.append("| --- " + "| --- " * len(cols_t3) + "|")
    for m in methods:
        cells = [m]
        for label, _ in cols_t3:
            r = results[label].get(m, {})
            if m == "Random":
                cells.append(f"{r.get('lpips', float('nan')):.3f} / -")
            else:
                cells.append(f"{r.get('lpips', float('nan')):.3f} / "
                             f"{r.get('same_mix_acc', float('nan')):.2f}")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n_LPIPS computed with LPIPS-AlexNet on 25 paired images per model._")
    text = "\n".join(md)
    with open("/home/haoming/finexl/REPRODUCTION_RESULTS.md", "w") as f:
        f.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
