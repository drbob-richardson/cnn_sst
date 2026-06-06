"""The showcase: input-mask attribution on the WeatherBench 2-day tendency task.

This task is genuinely nonlinear (the CNN beats logistic), so the input mask should now be
IDENTIFIABLE (unlike near-linear ENSO, where a flexible head routed around it). We train the
gated CNN (pixel / 2x2 / 4x4 masks) plus L0 / STG / attention gates, and post-hoc attributions
on the ungated model, and assess:

  - predictive AUROC (gating should not hurt),
  - mask CONCENTRATION (std of the mask; does it localize or go uniform?),
  - AGREEMENT with a per-cell correlation reference (field at t vs the future change),
  - a global map figure (mask + post-hoc + reference) with the target box marked.

Runs on the local Z500 cache. Usage:  .venv/bin/python masking/wb_mask.py --lead-days 2 --seeds 2
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
import torch.nn as nn
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam, LayerAttribution
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.sst_masking import MaskGate  # noqa: E402
from masking.sim_attribution_benchmark import L0Gate, STGGate, SpatialAttnGate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results" / "wb_z500.npz"
WRITEUP = REPO / "write_up"
TGT_LAT, TGT_LON = (40, 58), (200, 245)
LAM_SP, LAM_TV = 1e-2, 5e-2


class GatedFlatCNN(nn.Module):
    def __init__(self, H, W, gate=None, channels=(16, 32)):
        super().__init__()
        self.gate = gate
        chs = [1] + list(channels); layers = []
        for i in range(len(chs) - 1):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            fd = self.features(torch.zeros(1, 1, H, W)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fd, 1))

    def forward(self, x):
        if self.gate is not None:
            x, _ = self.gate(x)
        return self.head(self.features(x).flatten(1))


def gate_penalty(g):
    if isinstance(g, MaskGate):
        return g.penalty(LAM_SP, LAM_TV)
    if isinstance(g, (L0Gate, STGGate)):
        return 1e-2 * g.reg_loss()
    return torch.tensor(0.0)


def gate_imap(g, Xte_t):
    if isinstance(g, MaskGate):
        return g.importance()
    if isinstance(g, SpatialAttnGate):
        return g.importance_map(Xte_t)
    return g.importance_map()


def fit(make_gate, X, y, a, b, seed, epochs=40, bs=64):
    H, W = X.shape[2], X.shape[3]
    Xtr = torch.tensor(X[:a]); ytr = torch.tensor(y[:a]).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    gate = make_gate(H, W) if make_gate else None
    model = GatedFlatCNN(H, W, gate)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    best = {"va": -1, "auroc": float("nan"), "imap": None, "model": model}
    for ep in range(epochs):
        if hasattr(gate, "tau"):
            gate.tau = 2.0 - 1.99 * (ep / (epochs - 1))
        model.train()
        for i in torch.randperm(a).split(bs):
            loss = bce(model(Xtr[i]), ytr[i]) + (gate_penalty(gate) if gate is not None else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best["va"]:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte))).numpy().ravel()
            best = {"va": va, "auroc": roc_auc_score(yte, pt), "model": model,
                    "imap": None if gate is None else gate_imap(gate, torch.tensor(Xte[:400]))}
    return best


def posthoc(model, Xte, ref):
    model.eval(); inp = torch.tensor(Xte, requires_grad=True); base = torch.zeros_like(inp)
    out = {}
    for nm, fn in [("saliency", lambda: Saliency(model).attribute(inp, target=0)),
                   ("integrated_grad", lambda: IntegratedGradients(model).attribute(inp, baselines=base, target=0)),
                   ("gradient_shap", lambda: GradientShap(model).attribute(inp, baselines=base, target=0, n_samples=8, stdevs=0.1))]:
        a = fn().detach().abs().mean(dim=tuple(range(inp.dim() - 2))).numpy()
        out["posthoc_" + nm] = (a, float(np.corrcoef(a.ravel(), ref.ravel())[0, 1]))
    lc = [m for m in model.features if isinstance(m, nn.Conv2d)][-1]
    gc = LayerGradCam(model, lc).attribute(inp, target=0)
    gc = LayerAttribution.interpolate(gc, Xte.shape[2:]).detach().relu().mean(dim=(0, 1)).numpy()
    out["posthoc_grad_cam"] = (gc, float(np.corrcoef(gc.ravel(), ref.ravel())[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-days", type=int, default=2); ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    d = np.load(CACHE); z, month, lat, lon = d["z"], d["month"], d["lat"], d["lon"]
    T = len(z); ntr_clim = int(0.8 * T)
    clim = np.stack([z[:ntr_clim][month[:ntr_clim] == m].mean(0) for m in range(1, 13)])
    anom = (z - clim[month - 1]).astype(np.float32)
    li = np.where((lat >= TGT_LAT[0]) & (lat <= TGT_LAT[1]))[0]
    lj = np.where((lon >= TGT_LON[0]) & (lon <= TGT_LON[1]))[0]
    tgt = anom[:, li][:, :, lj].mean(axis=(1, 2))
    k = args.lead_days * 4
    dy = (tgt[k:] - tgt[:len(tgt) - k]).astype(np.float32)            # tendency
    X = anom[:len(dy), None]
    N = len(dy); a, b = int(0.6 * N), int(0.8 * N)
    thr = np.quantile(dy[:int(0.8 * N)], 2 / 3)
    y = (dy > thr).astype(np.float32)
    # per-cell correlation reference: field at t vs the (continuous) future change, on train
    Xc = X[:a, 0].reshape(a, -1); dyc = dy[:a]
    ref = (np.corrcoef(np.vstack([Xc.T, dyc]))[-1, :-1]).reshape(X.shape[2], X.shape[3])
    refabs = np.abs(ref)
    print(f"WeatherBench 2-day tendency: N={N}, grid {X.shape[2]}x{X.shape[3]}, event-rate={y.mean():.2f}")

    GATES = {"ungated": None, "ours_pixel": lambda H, W: MaskGate(H, W, 1),
             "ours_2x2": lambda H, W: MaskGate(H, W, 2), "ours_4x4": lambda H, W: MaskGate(H, W, 4),
             "L0_gate": lambda H, W: L0Gate(H, W), "STG_gate": lambda H, W: STGGate(H, W),
             "attention": lambda H, W: SpatialAttnGate(H, W)}
    rows, maps, t0 = [], {}, time.time()
    ung = None
    for nm, mk in GATES.items():
        aus, imaps = [], []
        for s in range(args.seeds):
            r = fit(mk, X, y, a, b, s)
            aus.append(r["auroc"])
            if nm == "ungated" and s == 0:
                ung = r["model"]
            if r["imap"] is not None:
                imaps.append(r["imap"])
        rec = {"method": nm, "auroc": np.mean(aus), "auroc_sd": np.std(aus)}
        if imaps:
            avg = np.mean(imaps, 0); maps[nm] = avg
            rec["agree"] = float(np.corrcoef(avg.ravel(), refabs.ravel())[0, 1])
            rec["concentration"] = float(avg.std())
        rows.append(rec)
        print(f"  {nm:<11} AUROC={np.mean(aus):.3f}±{np.std(aus):.3f}"
              + (f"  agree={rec['agree']:+.3f}  concentration(std)={rec['concentration']:.3f}" if imaps else ""))
    ph = posthoc(ung, X[b:b + 400], refabs)
    for nm, (m, ag) in ph.items():
        maps[nm] = m; print(f"  {nm:<16} agree={ag:+.3f}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(REPO / "results" / "wb_mask.csv", index=False)

    # figure: reference + ours_pixel + a couple post-hoc, with target box
    def draw(ax, M, title):
        ax.imshow(M, cmap="viridis", aspect="auto", origin="lower",
                  extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())])
        ax.add_patch(plt.Rectangle((TGT_LON[0], TGT_LAT[0]), TGT_LON[1] - TGT_LON[0], TGT_LAT[1] - TGT_LAT[0],
                                   fill=False, edgecolor="red", lw=1.5))
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    panels = [(refabs, "Correlation ref."), (maps.get("ours_pixel"), "Pixel mask"),
              (maps.get("ours_2x2"), "2x2 mask"), (maps.get("posthoc_integrated_grad"), "Integrated grad"),
              (maps.get("posthoc_grad_cam"), "Grad-CAM"), (maps.get("attention"), "Attention")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 5))
    for ax, (M, t) in zip(axes.flat, panels):
        draw(ax, M if M is not None else np.zeros_like(refabs), t)
    fig.suptitle("WeatherBench 2-day tendency: attribution maps (red box = target region)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(WRITEUP / "wb_mask_maps.png", dpi=200, bbox_inches="tight")
    print(f"\nTotal {time.time()-t0:.0f}s -> results/wb_mask.csv, write_up/wb_mask_maps.png")


if __name__ == "__main__":
    main()
