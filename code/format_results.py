"""Format selection-task results into a readable Markdown table per the
paper's Table 1/2 layout."""
import json, os, sys, glob

ROOT = "/home/haoming/finexl/runs/results"
out = "/home/haoming/finexl/results/REPRODUCTION.md"

def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "sel_*"))):
        rj = os.path.join(d, "results.json")
        if not os.path.exists(rj):
            continue
        levels = os.path.basename(d).replace("sel_", "").replace("_", ", ")
        n_levels = len(levels.split(","))
        r = load(rj)
        rows.append((n_levels, levels, r))

    rows.sort(key=lambda kv: kv[0])

    md = []
    md.append("## Reproduction: Table 1 (single-aspect personalization, artwork)\n")
    md.append("Selection-task LPIPS (lower = better) and exact-match accuracy.\n")
    md.append("Pool size = 4 styles × N levels.\n")
    md.append("| Method | " + " | ".join(
        f"{n_levels} levels  LPIPS↓ / Acc↑" for n_levels, _, _ in rows) + " |")
    md.append("|---" * (len(rows) + 1) + "|")
    methods = ["Random", "Naive", "GSCLIP", "VisDiff", "FineXL"]
    for m in methods:
        cells = [m]
        for _, _, r in rows:
            if m not in r:
                cells.append("--")
            else:
                rec = r[m]
                if "acc" in rec:
                    cells.append(f"{rec['lpips']:.3f} / {rec['acc']:.2f}")
                else:
                    cells.append(f"{rec['lpips']:.3f} / --")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"_Selection-task levels per column: {[r[1] for r in rows]}_")
    text = "\n".join(md)
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
