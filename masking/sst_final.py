"""Final, defensible SST analysis on the redeemed backbone.

Backbone chosen by the architecture ablation (masking/sst_cnn_ablation.py): a compact CNN with a
LOCATION-PRESERVING flatten head (two 3x3 convs [16,32] + max-pool, batch normalization, ReLU,
flatten -> dropout -> linear), trained with batch 16, no input dropout, Adam 1e-3, ~120 epochs
early-stopped on validation AUROC. This matches the linear logistic ceiling, unlike the inherited
global-average-pooling backbone.

Same 3-way temporal split (train 60% / val 20% early-stop / test 20%). Reports, on the held-out
test block:
  - predictive AUROC: logistic + gradient boosting + the CNN with none/pixel/2x2/4x4/L0/STG/attn,
  - attribution agreement with the lagged-correlation reference: in-model gates + post-hoc methods,
and regenerates write_up/sst_masks_vs_reference.png for the final backbone.

Usage:  .venv/bin/python masking/sst_final.py --leads 3 6 --seeds 3
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_masking import MaskGate  # noqa: E402
from masking.sim_attribution_benchmark import L0Gate, STGGate, SpatialAttnGate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
DATA = REPO_ROOT / "data" / "sst_data.nc"
EXTENT = [170, 260, -15, 15]
DEVICE = torch.device("cpu")
BS, EPOCHS = 16, 120


class FinalCNN(nn.Module):
    """Compact CNN with a location-preserving flatten head."""
    def __init__(self, channels=(16, 32), gate=None, H=30, W=90):
        super().__init__()
        self.gate = gate
        chs = [1] + list(channels)
        layers = []
        for i in range(len(chs) - 1):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]),
                       nn.ReLU(), nn.MaxPool2d(2)]
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
        return g.penalty(1e-2, 0.2)
    if isinstance(g, (L0Gate, STGGate)):
        return 1e-2 * g.reg_loss()
    return torch.tensor(0.0)


def gate_imap(g, Xte_t):
    if isinstance(g, MaskGate):
        return g.importance()
    if isinstance(g, SpatialAttnGate):
        return g.importance_map(Xte_t)
    return g.importance_map()


def fit_cnn(make_gate, lead, seed):
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    H, W = X.shape[2], X.shape[3]
    T = len(y); a, b = int(0.6 * T), int(0.8 * T)
    Xtr = torch.tensor(X[:a], dtype=torch.float32); ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    gate = make_gate(H, W) if make_gate else None
    model = FinalCNN(gate=gate, H=H, W=W).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    best = {"va": -1, "test_auroc": float("nan"), "imap": None, "model": model}
    for ep in range(EPOCHS):
        if hasattr(gate, "tau"):
            gate.tau = 2.0 - 1.99 * (ep / (EPOCHS - 1))
        model.train()
        for i in torch.randperm(a).split(BS):
            xb = Xtr[i]
            loss = bce(model(xb), ytr[i]) + (gate_penalty(gate) if gate is not None else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best["va"]:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
            best = {"va": va, "test_auroc": roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan"),
                    "imap": None if gate is None else gate_imap(gate, torch.tensor(Xte, dtype=torch.float32)),
                    "model": model, "Xte": X[b:]}
    return best


def posthoc(model, Xte, corr, H, W):
    model.eval()
    inp = torch.tensor(Xte, dtype=torch.float32, requires_grad=True)
    base = torch.zeros_like(inp)
    out = {}
    for name, fn in [("saliency", lambda: Saliency(model).attribute(inp, target=0)),
                     ("integrated_grad", lambda: IntegratedGradients(model).attribute(inp, baselines=base, target=0)),
                     ("gradient_shap", lambda: GradientShap(model).attribute(inp, baselines=base, target=0, n_samples=8, stdevs=0.1))]:
        a = fn().detach().abs().mean(dim=tuple(range(inp.dim() - 2))).numpy()
        out["posthoc_" + name] = float(np.corrcoef(a.ravel(), corr.ravel())[0, 1])
    last_conv = [m for m in model.features if isinstance(m, nn.Conv2d)][-1]
    gc = LayerGradCam(model, last_conv).attribute(inp, target=0)
    gc = LayerAttribution.interpolate(gc, (H, W)).detach().relu().mean(dim=(0, 1)).numpy()
    out["posthoc_grad_cam"] = float(np.corrcoef(gc.ravel(), corr.ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    import pandas as pd
    t0 = time.time()
    pred, attr, maps = [], [], {}
    GATES = {"ungated": None, "ours_pixel": lambda H, W: MaskGate(H, W, 1),
             "ours_2x2": lambda H, W: MaskGate(H, W, 2), "ours_4x4": lambda H, W: MaskGate(H, W, 4),
             "L0_gate": lambda H, W: L0Gate(H, W), "STG_gate": lambda H, W: STGGate(H, W),
             "attention": lambda H, W: SpatialAttnGate(H, W)}

    for lead in args.leads:
        X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
        H, W = X.shape[2], X.shape[3]; T = len(y); b = int(0.8 * T)
        corr = np.abs(ref[f"lead{lead}_corr"])
        Xf = X.reshape(T, -1); sc = StandardScaler().fit(Xf[:b])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf[:b]), y[:b])
        pred.append({"lead": lead, "method": "logistic (linear)", "auroc": roc_auc_score(y[b:], lr.predict_proba(sc.transform(Xf[b:]))[:, 1]), "sd": 0.0})
        hgb = [roc_auc_score(y[b:], HistGradientBoostingClassifier(random_state=s, max_iter=200).fit(Xf[:b], y[:b]).predict_proba(Xf[b:])[:, 1]) for s in range(args.seeds)]
        pred.append({"lead": lead, "method": "gradient boosting", "auroc": np.mean(hgb), "sd": np.std(hgb)})

        ungated0 = None
        for mname, mk in GATES.items():
            aus, imaps = [], []
            for s in range(args.seeds):
                r = fit_cnn(mk, lead, s)
                aus.append(r["test_auroc"])
                if mname == "ungated" and s == 0:
                    ungated0 = (r["model"], r["Xte"])
                if r["imap"] is not None:
                    imaps.append(r["imap"])
            pred.append({"lead": lead, "method": "CNN_" + mname, "auroc": np.mean(aus), "sd": np.std(aus)})
            if imaps:
                avg = np.mean(imaps, 0); maps[f"lead{lead}_{mname}"] = avg
                attr.append({"lead": lead, "method": mname, "agree": float(np.corrcoef(avg.ravel(), corr.ravel())[0, 1])})
        for k, v in posthoc(ungated0[0], ungated0[1], corr, H, W).items():
            attr.append({"lead": lead, "method": k, "agree": v})
        print(f"  lead {lead} done ({time.time()-t0:.0f}s)")

    pd.DataFrame(pred).to_csv(REPO_ROOT / "results" / "sst_final_predictive.csv", index=False)
    pd.DataFrame(attr).to_csv(REPO_ROOT / "results" / "sst_final_attribution.csv", index=False)
    np.savez(REPO_ROOT / "results" / "sst_final_maps.npz", **maps)

    # regenerate mask-vs-reference figure for the final backbone
    cols = ["corr", "ours_pixel", "ours_2x2", "ours_4x4"]
    titles = {"corr": "Correlation ref.", "ours_pixel": "Pixel mask", "ours_2x2": "2x2 tiles", "ours_4x4": "4x4 tiles"}
    fig, axes = plt.subplots(len(args.leads), 4, figsize=(13, 3.0 * len(args.leads)), squeeze=False)
    for r, lead in enumerate(args.leads):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            m = np.abs(ref[f"lead{lead}_corr"]) if key == "corr" else maps.get(f"lead{lead}_{key}")
            if m is not None:
                ax.imshow(m, cmap="viridis", extent=EXTENT, aspect="auto", origin="upper")
            ax.set_title(f"{titles[key]} — lead {lead}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Learned masks vs. correlation reference (location-preserving flatten CNN)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(WRITEUP / "sst_masks_vs_reference.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    pdf = pd.DataFrame(pred); adf = pd.DataFrame(attr)
    for lead in args.leads:
        print(f"\n=== Lead {lead}: PREDICTIVE (test AUROC) ===")
        for _, r in pdf[pdf.lead == lead].sort_values("auroc", ascending=False).iterrows():
            print(f"  {r['method']:<16} {r['auroc']:.3f} ± {r['sd']:.3f}")
        print(f"--- Lead {lead}: ATTRIBUTION agreement ---")
        for _, r in adf[adf.lead == lead].sort_values("agree", ascending=False).iterrows():
            print(f"  {r['method']:<20} {r['agree']:+.3f}")
    print(f"\nTotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
