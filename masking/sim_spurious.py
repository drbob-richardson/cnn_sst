"""Spurious-correlation simulation regime (the key scenario from Sim Study 2).

Setup: a CAUSAL region (two blobs A, B; XOR of their intensities sets the label) and a
SPURIOUS region S that is *correlated* with the label (its sign encodes a noisily-flipped
copy of y, with random magnitude) but is **not causal**. A model can take the S shortcut
to ~(1-flip_p) accuracy, or learn the harder causal XOR.

Question: which attribution method reveals reliance on the spurious region, and does the
jointly-trained TV+sparsity mask suppress it? We score each method's importance map by
overlap with the CAUSAL region (higher = better) and the SPURIOUS region (lower = better).

Emits two paper figures into write_up/:
  - sim_spurious_attributions.png : per-method attribution maps with causal (green) /
    spurious (red dashed) contours.
  - sim_spurious_iou.png          : IoU(causal) vs IoU(spurious) per method, mean +/- SD.

Usage (from repo root):
    .venv/bin/python masking/sim_spurious.py --seeds 15 --epochs 60
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.sim_attribution_benchmark import (  # noqa: E402
    L0Gate, STGGate, SpatialAttnGate, train_gated, train_ours, posthoc_map, DEVICE,
)
from masking.simulation_study_1 import (  # noqa: E402
    make_circle_mask, region_A, region_B, generate_labels,
    SmallCNN, PixelMaskGate, compute_interpretability_metrics, eval_metrics, H, W, N,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
# Spurious region: a blob away from the causal blobs A(10,15) & B(14,30).
region_S = make_circle_mask(H, W, center=(6, 38), radius=4)

POSTHOC = [("saliency", "saliency", Saliency),
           ("integrated_grad", "ig", IntegratedGradients),
           ("grad_cam", "gradcam", LayerGradCam),
           ("gradient_shap", "gshap", GradientShap)]
PANEL_ORDER = ["saliency", "integrated_grad", "grad_cam", "gradient_shap",
               "l0_gate", "stg_gate", "spatial_attn", "ours_TV"]
NICE = {"saliency": "Saliency", "integrated_grad": "Integrated Grad", "grad_cam": "Grad-CAM",
        "gradient_shap": "Gradient SHAP", "l0_gate": "L0 gate", "stg_gate": "STG gate",
        "spatial_attn": "Spatial attn", "ours_TV": "Ours (TV mask)"}


def simulate_spurious(n, flip_p=0.25, s_scale=1.5):
    X = (np.random.randn(n, 1, H, W) * 0.5).astype(np.float32)
    cA = np.random.randn(n, 1, 1, 1).astype(np.float32)
    cB = np.random.randn(n, 1, 1, 1).astype(np.float32)
    X += cA * region_A[None, None] - cB * region_B[None, None]
    y, _ = generate_labels(X, region_A, region_B)                  # XOR over A, B (causal)
    flip = (np.random.rand(n) < flip_p)
    s_sign = np.where(flip, 1 - y, y)                              # spurious: noisily-flipped label
    mag = (np.random.rand(n) * s_scale).astype(np.float32)
    sval = ((2 * s_sign - 1) * mag).astype(np.float32)
    X += sval[:, None, None, None] * region_S[None, None]
    return X.astype(np.float32), y.astype(np.float32)


def run_seed(seed, epochs, n_attr, collect_maps=False):
    np.random.seed(seed); torch.manual_seed(seed)
    X, y = simulate_spurious(N)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    ntr = int(0.8 * N)
    tr, te = random_split(ds, [ntr, N - ntr])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    te_loader = DataLoader(te, batch_size=128)
    Xte = torch.stack([te[i][0] for i in range(len(te))]).float()[:n_attr]

    # signal = causal (A,B); distractor = spurious (S)
    regions = (region_A, region_B, region_S, region_S)
    rows, maps = [], {}

    def record(method, imap, model):
        m = compute_interpretability_metrics(imap, *regions)
        acc, f1 = eval_metrics(model, te_loader)
        rows.append({"seed": seed, "method": method,
                     "iou_causal": m["iou_true"], "iou_spurious": m["iou_distractor"],
                     "mass_causal": m["mass_true"], "mass_spurious": m["mass_distractor"],
                     "acc": acc, "f1": f1})
        if collect_maps:
            maps[method] = imap

    base = train_gated(SmallCNN().to(DEVICE), tr_loader, epochs)
    for name, kind, ctor in POSTHOC:
        method = ctor(base, base.conv2) if kind == "gradcam" else ctor(base)
        record(name, posthoc_map(method, Xte, kind), base)

    for name, ctor, lam in [("l0_gate", lambda: L0Gate(H, W), 1.0),
                            ("stg_gate", lambda: STGGate(H, W), 1.0),
                            ("spatial_attn", lambda: SpatialAttnGate(H, W), 0.0)]:
        g = ctor().to(DEVICE)
        mdl = train_gated(SmallCNN(g).to(DEVICE), tr_loader, epochs, g, lam, tv_lambda=5e-2)
        record(name, g.importance_map(Xte), mdl)

    g = PixelMaskGate(H, W).to(DEVICE)
    mdl = train_ours(SmallCNN(g).to(DEVICE), g, tr_loader, epochs, lambda_tv=5e-2)
    imap = torch.sigmoid(g.z_main / g.tau).detach().view(H, W).cpu().numpy()
    record("ours_TV", imap, mdl)

    if collect_maps:
        maps["_input_example"] = X[int(np.argmax(y))][0]   # a y=1 example
    return rows, maps


def _overlay(ax, img, title, norm=True):
    a = (img - img.min()) / (img.max() - img.min() + 1e-8) if norm else img
    ax.imshow(a, cmap="hot", aspect="auto")
    ax.contour(np.logical_or(region_A, region_B), levels=[0.5], colors="lime", linewidths=1.2)
    ax.contour(region_S, levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def make_panel(maps, path):
    fig, axes = plt.subplots(3, 3, figsize=(9, 5.2))
    ex = maps["_input_example"]
    ax = axes.flat[0]
    ax.imshow(ex, cmap="gray", aspect="auto")
    ax.contour(np.logical_or(region_A, region_B), levels=[0.5], colors="lime", linewidths=1.2)
    ax.contour(region_S, levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
    ax.set_title("Input (causal=green, spurious=red)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax, name in zip(axes.flat[1:], PANEL_ORDER):
        _overlay(ax, maps[name], NICE[name])
    fig.suptitle("Attribution under spurious correlation (causal = green, spurious = red dashed)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def make_bar(df, path):
    methods = PANEL_ORDER
    causal = [df[df.method == m]["iou_causal"].mean() for m in methods]
    spur = [df[df.method == m]["iou_spurious"].mean() for m in methods]
    cerr = [df[df.method == m]["iou_causal"].std() for m in methods]
    serr = [df[df.method == m]["iou_spurious"].std() for m in methods]
    x = np.arange(len(methods)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(x - w / 2, causal, w, yerr=cerr, capsize=3, label="Causal region (higher better)", color="#2c7fb8")
    ax.bar(x + w / 2, spur, w, yerr=serr, capsize=3, label="Spurious region (lower better)", color="#d95f0e")
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IoU with region (top-10%)"); ax.legend(fontsize=8)
    ax.set_title("Spurious-correlation regime: causal vs spurious attribution overlap")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-attr", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/sim_spurious.csv")
    args = ap.parse_args()

    t0 = time.time()
    rows, maps0 = [], None
    for s in range(args.seeds):
        r, m = run_seed(s, args.epochs, args.n_attr, collect_maps=(s == 0))
        rows += r
        if s == 0:
            maps0 = m
        print(f"  seed {s} done")

    import pandas as pd
    df = pd.DataFrame(rows)
    (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPO_ROOT / args.out, index=False)

    make_panel(maps0, WRITEUP / "sim_spurious_attributions.png")
    make_bar(df, WRITEUP / "sim_spurious_iou.png")

    print(f"\n=== Spurious-correlation regime: mean over {args.seeds} seeds ===")
    print(f"{'method':<16}{'IoU_causal':>11}{'IoU_spur':>10}{'mass_caus':>10}{'mass_spur':>10}{'acc':>7}{'f1':>7}")
    for m in PANEL_ORDER:
        sub = df[df.method == m]
        print(f"{m:<16}{sub['iou_causal'].mean():>11.3f}{sub['iou_spurious'].mean():>10.3f}"
              f"{sub['mass_causal'].mean():>10.3f}{sub['mass_spurious'].mean():>10.3f}"
              f"{sub['acc'].mean():>7.3f}{sub['f1'].mean():>7.3f}")
    print(f"\nFigures -> {WRITEUP/'sim_spurious_attributions.png'} , {WRITEUP/'sim_spurious_iou.png'}")
    print(f"CSV -> {REPO_ROOT/args.out}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
