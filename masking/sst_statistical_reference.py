"""Statistical reference attribution for the El Nino / SST application.

Provides two principled, tractable "where is the predictive signal" maps that the learned
mask can be validated against (convergent validity, since the application has no ground truth):

  1. Lagged point-biserial CORRELATION map: per-cell correlation between the SST anomaly and
     the future El Nino label. The classic climate-statistics reference; model-free.
  2. FUSED-LASSO logistic regression: predict the label from the whole SST field with an
     L1 (sparsity) + total-variation (smoothness) penalty on the per-cell coefficients. This is
     the *linear statistical sibling* of our TV+sparsity mask (fused lasso == TV+L1 on
     coefficients), so agreement between the two is a meaningful sanity check.

Run on a spatially down-sampled field (the fused-lasso problem is otherwise p >> n at 0.25 deg).
Saves coefficient maps (npz) + a figure to write_up/sst_statistical_reference.png.

Usage:  .venv/bin/python masking/sst_statistical_reference.py --leads 3 6 --resolution 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
DATA = REPO_ROOT / "data" / "sst_data.nc"
EXTENT = [170, 260, -15, 15]   # lon/lat box of the analysis region (for display)


def correlation_map(X, y):
    """Per-cell point-biserial correlation between SST anomaly and the binary label."""
    x = X[:, 0]                                   # (T, H, W)
    xc = x - x.mean(0, keepdims=True)
    yc = (y - y.mean())
    num = (xc * yc[:, None, None]).sum(0)
    den = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum() + 1e-12)
    return num / den                              # (H, W), signed


def fused_lasso_logistic(X, y, l1=8e-2, ltv=6e-1, l2=1e-2, steps=2000, lr=0.05):
    """Logistic regression over the flattened field with L1 + TV (+ small L2) penalty on
    coefficients. Strong penalties are required because the problem is p >> n
    (e.g. 2,700 coefficients vs. ~500 samples); weak regularization just overfits noise."""
    T, _, H, W = X.shape
    Xf = torch.tensor(X.reshape(T, -1), dtype=torch.float32)
    Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-6)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(H * W, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        logits = Xf @ w + b
        wm = w.view(H, W)
        tv = (wm[:, 1:] - wm[:, :-1]).abs().sum() + (wm[1:, :] - wm[:-1, :]).abs().sum()
        loss = bce(logits, yt) + l1 * w.abs().mean() + ltv * tv / (H * W) + l2 * (w ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().view(H, W).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--resolution", type=int, default=4, help="spatial downsample for the field")
    args = ap.parse_args()

    maps = {}
    fig, axes = plt.subplots(len(args.leads), 2, figsize=(9, 3.0 * len(args.leads)), squeeze=False)
    for r, lead in enumerate(args.leads):
        X, y = process_data_multi_res(lead, resolution=args.resolution, file_path=DATA)
        print(f"lead {lead}: X={X.shape}, event rate={y.mean():.3f}")
        corr = correlation_map(X, y)
        coef = fused_lasso_logistic(X, y)
        maps[f"lead{lead}_corr"] = corr
        maps[f"lead{lead}_fusedlasso"] = coef
        for c, (m, title) in enumerate([(corr, "Lagged correlation"), (coef, "Fused-lasso logistic")]):
            ax = axes[r, c]
            v = np.abs(m).max() + 1e-9
            im = ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, extent=EXTENT, aspect="auto", origin="upper")
            ax.set_title(f"{title} — lead {lead} mo", fontsize=9)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            fig.colorbar(im, ax=ax, fraction=0.025)
    fig.suptitle("Statistical reference attribution for El Niño (signed)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = WRITEUP / "sst_statistical_reference.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    np.savez(REPO_ROOT / "results" / "sst_statistical_reference.npz", **maps)
    print(f"wrote {out_png}")
    print(f"wrote {REPO_ROOT/'results'/'sst_statistical_reference.npz'}")


if __name__ == "__main__":
    main()
