"""Generate the Simulation-Study-1 paper figures from our benchmark + sweep results.

Produces (into write_up/):
  - sim1_attributions.png : per-method attribution maps on a Sim-1 example, with
    signal (green) and distractor (red dashed) contours (retrains seed 0).
  - sim1_faithfulness.png : IoU(signal) vs IoU(distractor) per method, mean +/- SD
    (from results/sim_attribution_benchmark.csv).
  - sim1_frontier.png     : faithfulness x accuracy frontier (IoU_signal vs F1) across
    lambda for L0 / STG / ours (from results/sim_lambda_sweep.csv).

Usage:  .venv/bin/python masking/make_sim1_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.sim_attribution_benchmark import (  # noqa: E402
    L0Gate, STGGate, SpatialAttnGate, train_gated, train_ours, posthoc_map, DEVICE,
)
from masking.simulation_study_1 import (  # noqa: E402
    simulate_spatial_input, generate_labels, signal_regions,
    region_A, region_B, region_distractor_1, region_distractor_2,
    SmallCNN, PixelMaskGate, H, W, N,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
SIGNAL = np.logical_or(region_A, region_B)
DISTRACT = np.logical_or(region_distractor_1, region_distractor_2)
PANEL = ["saliency", "integrated_grad", "grad_cam", "gradient_shap",
         "l0_gate", "stg_gate", "spatial_attn", "ours_TV"]
NICE = {"saliency": "Saliency", "integrated_grad": "Integrated Grad", "grad_cam": "Grad-CAM",
        "gradient_shap": "Gradient SHAP", "l0_gate": "L0 gate", "stg_gate": "STG gate",
        "spatial_attn": "Spatial attn", "ours_TV": "Ours (TV mask)"}


def collect_maps(epochs=60, n_attr=100, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    X = simulate_spatial_input(N, H, W, signal_regions)
    y, _ = generate_labels(X, region_A, region_B)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    tr, te = random_split(ds, [int(0.8 * N), N - int(0.8 * N)])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    Xte = torch.stack([te[i][0] for i in range(len(te))]).float()[:n_attr]
    maps = {"_input": X[int(np.argmax(y))][0]}

    base = train_gated(SmallCNN().to(DEVICE), tr_loader, epochs)
    for name, kind, ctor in [("saliency", "saliency", Saliency),
                             ("integrated_grad", "ig", IntegratedGradients),
                             ("grad_cam", "gradcam", LayerGradCam),
                             ("gradient_shap", "gshap", GradientShap)]:
        method = ctor(base, base.conv2) if kind == "gradcam" else ctor(base)
        maps[name] = posthoc_map(method, Xte, kind)
    for name, ctor, lam in [("l0_gate", lambda: L0Gate(H, W), 1.0),
                            ("stg_gate", lambda: STGGate(H, W), 1.0),
                            ("spatial_attn", lambda: SpatialAttnGate(H, W), 0.0)]:
        g = ctor().to(DEVICE)
        train_gated(SmallCNN(g).to(DEVICE), tr_loader, epochs, g, lam, tv_lambda=5e-2)
        maps[name] = g.importance_map(Xte)
    g = PixelMaskGate(H, W).to(DEVICE)
    train_ours(SmallCNN(g).to(DEVICE), g, tr_loader, epochs, lambda_tv=5e-2)
    maps["ours_TV"] = torch.sigmoid(g.z_main / g.tau).detach().view(H, W).cpu().numpy()
    return maps


def fig_panel(maps, path):
    fig, axes = plt.subplots(3, 3, figsize=(9, 5.2))
    ax = axes.flat[0]
    ax.imshow(maps["_input"], cmap="gray", aspect="auto")
    ax.contour(SIGNAL, [0.5], colors="lime", linewidths=1.2)
    ax.contour(DISTRACT, [0.5], colors="red", linewidths=1.2, linestyles="--")
    ax.set_title("Input (signal=green, distractor=red)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax, name in zip(axes.flat[1:], PANEL):
        a = maps[name]; a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        ax.imshow(a, cmap="hot", aspect="auto")
        ax.contour(SIGNAL, [0.5], colors="lime", linewidths=1.0)
        ax.contour(DISTRACT, [0.5], colors="red", linewidths=1.0, linestyles="--")
        ax.set_title(NICE[name], fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Attribution maps on a structured-signal example (signal = green, distractor = red dashed)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_faithfulness(path):
    df = pd.read_csv(REPO_ROOT / "results/sim_attribution_benchmark.csv")
    sig = [df[df.method == m]["iou_true"].mean() for m in PANEL]
    dis = [df[df.method == m]["iou_distractor"].mean() for m in PANEL]
    se = [df[df.method == m]["iou_true"].std() for m in PANEL]
    de = [df[df.method == m]["iou_distractor"].std() for m in PANEL]
    x = np.arange(len(PANEL)); w = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(x - w / 2, sig, w, yerr=se, capsize=3, color="#2c7fb8", label="Signal IoU (higher better)")
    ax.bar(x + w / 2, dis, w, yerr=de, capsize=3, color="#d95f0e", label="Distractor IoU (lower better)")
    ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in PANEL], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IoU with region (top-10%)"); ax.legend(fontsize=8)
    ax.set_title("Attribution faithfulness vs. ground truth (15 seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_frontier(path):
    df = pd.read_csv(REPO_ROOT / "results/sim_lambda_sweep.csv")
    df = df[df.tv == 0.05]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    styles = {"l0": ("#1b9e77", "o", "L0 gate"), "stg": ("#7570b3", "s", "STG gate"),
              "ours": ("#d95f02", "^", "Ours (TV mask)")}
    for m, (c, mk, lab) in styles.items():
        g = df[df.method == m].groupby("lam").agg(f1=("f1", "mean"), iou=("iou_true", "mean")).reset_index()
        g = g.sort_values("f1")
        ax.plot(g.f1, g.iou, marker=mk, color=c, label=lab, ms=7, lw=1.5)
    ax.set_xlabel("Predictive F1 (higher better) →")
    ax.set_ylabel("Signal IoU (higher better) →")
    ax.set_title("Faithfulness × accuracy frontier (sweep over sparsity λ)")
    ax.legend(fontsize=9, loc="lower left")
    ax.annotate("ours reaches high F1\nL0/STG cannot", xy=(0.60, 0.72), xytext=(0.50, 0.55),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    print("training seed 0 for attribution panel ...")
    maps = collect_maps()
    fig_panel(maps, WRITEUP / "sim1_attributions.png")
    fig_faithfulness(WRITEUP / "sim1_faithfulness.png")
    fig_frontier(WRITEUP / "sim1_frontier.png")
    print("wrote sim1_attributions.png, sim1_faithfulness.png, sim1_frontier.png to write_up/")


if __name__ == "__main__":
    main()
