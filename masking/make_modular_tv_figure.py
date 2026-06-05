"""Figure for the modular-TV result: adding TV improves L0 / STG / ours faithfulness.

Reads results/sim_lambda_sweep.csv (no retraining) and plots signal-IoU vs sparsity weight
with TV off vs on, per gating mechanism --- showing TV lifts faithfulness (most when the base
gate is off its IoU ceiling). Writes write_up/sim1_modular_tv.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO_ROOT / "results" / "sim_lambda_sweep.csv")

styles = {"l0": ("#1b9e77", "o", "L0 gate"), "stg": ("#7570b3", "s", "STG gate"),
          "ours": ("#d95f02", "^", "Ours (sigmoid gate)")}
fig, ax = plt.subplots(figsize=(6.6, 4.6))
for m, (c, mk, lab) in styles.items():
    base = df[(df.method == m) & (df.tv == 0.0)].groupby("lam")["iou_true"].mean()
    tvon = df[(df.method == m) & (df.tv == 0.05)].groupby("lam")["iou_true"].mean()
    j = base.index.intersection(tvon.index)
    ax.scatter(base[j], (tvon[j] - base[j]), color=c, marker=mk, s=60, label=lab, zorder=3)
ax.axhline(0, color="gray", lw=1, ls=":")
ax.set_xlabel("Signal IoU without TV (each point = one sparsity setting)")
ax.set_ylabel(r"$\Delta$ Signal IoU from adding TV")
ax.set_title("Total variation improves faithfulness across learned gates\n"
             "(largest gains where the base gate localizes worst; vanishes at the ceiling)")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
out = REPO_ROOT / "write_up" / "sim1_modular_tv.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
