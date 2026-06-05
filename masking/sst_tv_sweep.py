"""Small, principled TV-weight sweep for the El Nino masks.

Selection criterion: the lambda_tv that makes the masks smoother and better aligned with the
lagged-correlation reference WITHOUT hurting predictive AUROC. Not an exhaustive search --- a
handful of values, the data picks. Produces a figure showing the mask smoothing with lambda_tv
next to the correlation reference, and a table of AUROC + agreement per setting.

Usage:  .venv/bin/python masking/sst_tv_sweep.py --leads 3 6 --seeds 2
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.sst_masking import train_eval, REPO_ROOT, WRITEUP, EXTENT  # noqa: E402

TV_GRID = [0.01, 0.05, 0.2, 0.5]
GATES = [("pixel", 1), ("tile_2x2", 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    rows = []
    pixel_maps = {}     # (lead, ltv) -> avg pixel mask, for the figure

    for lead in args.leads:
        corr = np.abs(ref[f"lead{lead}_corr"])
        for gname, tile in GATES:
            for ltv in TV_GRID:
                aurocs, f1s, maps = [], [], []
                for s in range(args.seeds):
                    m, mp = train_eval(lead, tile, s, args.epochs, lam_tv=ltv)
                    aurocs.append(m["auroc"]); f1s.append(m["f1"]); maps.append(mp)
                avg = np.mean(maps, 0)
                agree = float(np.corrcoef(avg.ravel(), corr.ravel())[0, 1])
                rows.append({"lead": lead, "gate": gname, "lam_tv": ltv,
                             "auroc": np.nanmean(aurocs), "f1": np.mean(f1s), "agree": agree})
                if gname == "pixel":
                    pixel_maps[(lead, ltv)] = avg
                print(f"  lead {lead} {gname:<9} tv={ltv:<5} auroc={np.nanmean(aurocs):.3f} agree={agree:+.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_tv_sweep.csv", index=False)

    # Figure: pixel mask vs lambda_tv, per lead, alongside the correlation reference.
    ncol = 1 + len(TV_GRID)
    fig, axes = plt.subplots(len(args.leads), ncol, figsize=(3.0 * ncol, 3.0 * len(args.leads)), squeeze=False)
    for r, lead in enumerate(args.leads):
        ax = axes[r, 0]
        ax.imshow(np.abs(ref[f"lead{lead}_corr"]), cmap="viridis", extent=EXTENT, aspect="auto", origin="upper")
        ax.set_title(f"Correlation ref. — lead {lead}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        for c, ltv in enumerate(TV_GRID):
            ax = axes[r, c + 1]
            ax.imshow(pixel_maps[(lead, ltv)], cmap="viridis", extent=EXTENT, aspect="auto", origin="upper")
            ag = df[(df.lead == lead) & (df.gate == "pixel") & (df.lam_tv == ltv)]["agree"].iloc[0]
            ax.set_title(f"pixel, $\\lambda_{{TV}}$={ltv} (agree {ag:+.2f})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pixel mask vs. TV weight (smoothing toward the equatorial correlation signal)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(WRITEUP / "sst_tv_sweep.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print("\n=== summary (AUROC should stay flat; pick highest agreement) ===")
    print(df.to_string(index=False))
    print(f"\nwrote results/sst_tv_sweep.csv, write_up/sst_tv_sweep.png")


if __name__ == "__main__":
    main()
