"""Full method comparison on the SST data --- a reference frame for prediction and attribution.

Same 3-way temporal split (train 60% / val 20% for early stopping / untouched test 20%) and the
selected compact backbone as masking/sst_model_select.py. We compare, on the held-out test block:

Predictive skill (test AUROC), to judge whether our model is near the achievable ceiling:
  - Linear: L2 logistic regression on the flattened field.
  - Nonlinear, non-CNN: histogram gradient boosting.
  - Nonlinear CNN: ungated baseline, and the L0 / STG / attention / our-mask gates.

Attribution agreement (Pearson of the importance map with |lagged-correlation reference|):
  - Post-hoc on the ungated CNN: saliency, integrated gradients, Grad-CAM, gradient SHAP.
  - In-model gates: L0, STG, attention, ours.

Usage:  .venv/bin/python masking/sst_methods_compare.py --leads 3 6 --seeds 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam, LayerAttribution
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_model_select import ConfigCNN, CANDIDATES  # noqa: E402
from masking.sst_masking import MaskGate  # noqa: E402
from masking.sim_attribution_benchmark import L0Gate, STGGate, SpatialAttnGate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "sst_data.nc"
DEVICE = torch.device("cpu")                 # small model; CPU is reliable for captum + gates
SELECTED = CANDIDATES["small_relu"]          # (channels, pool, head, act) chosen by sst_model_select


def splits(lead):
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    T = len(y); a, b = int(0.6 * T), int(0.8 * T)
    return X, y, a, b


def gate_penalty(gate):
    if isinstance(gate, MaskGate):
        return gate.penalty(1e-2, 0.2)
    if isinstance(gate, (L0Gate, STGGate)):
        return 1e-2 * gate.reg_loss()
    return torch.tensor(0.0)


def gate_imap(gate, Xte_t):
    if isinstance(gate, MaskGate):
        return gate.importance()
    if isinstance(gate, SpatialAttnGate):
        return gate.importance_map(Xte_t)
    return gate.importance_map()             # L0 / STG


def fit_cnn(make_gate, lead, seed, epochs=60):
    X, y, a, b = splits(lead)
    H, W = X.shape[2], X.shape[3]
    Xtr = torch.tensor(X[:a], dtype=torch.float32)
    ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    gate = make_gate(H, W) if make_gate else None
    model = ConfigCNN(*SELECTED, gate=gate).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    n, bs = a, 128
    best = {"auroc": -1, "state": None, "imap": None}
    for ep in range(epochs):
        if hasattr(gate, "tau"):
            gate.tau = 2.0 - 1.99 * (ep / max(1, epochs - 1))
        model.train()
        for i in torch.randperm(n).split(bs):
            xb = Xtr[i].to(DEVICE)
            xb = xb * (torch.rand_like(xb) > 0.20) / 0.80
            loss = bce(model(xb), ytr[i].to(DEVICE)) + gate_penalty(gate) if gate is not None else bce(model(xb), ytr[i].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32).to(DEVICE))).cpu().numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best["auroc"]:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE))).cpu().numpy().ravel()
            best = {"auroc": va,
                    "test_auroc": roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan"),
                    "imap": None if gate is None else gate_imap(gate, torch.tensor(Xte, dtype=torch.float32)),
                    "model": model}
    return best


def posthoc_agreements(model, Xte, corr, H, W):
    model.eval()
    inp = torch.tensor(Xte, dtype=torch.float32, requires_grad=True)
    base = torch.zeros_like(inp)
    last_conv = [m for m in model.features if isinstance(m, nn.Conv2d)][-1]
    out = {}
    for name, fn in [
        ("saliency", lambda: Saliency(model).attribute(inp, target=0)),
        ("integrated_grad", lambda: IntegratedGradients(model).attribute(inp, baselines=base, target=0)),
        ("gradient_shap", lambda: GradientShap(model).attribute(inp, baselines=base, target=0, n_samples=8, stdevs=0.1)),
    ]:
        a = fn().detach().abs().mean(dim=tuple(range(inp.dim() - 2))).numpy()
        out[name] = float(np.corrcoef(a.ravel(), corr.ravel())[0, 1])
    # Grad-CAM: attribute at the conv layer, upsample to input resolution
    gc = LayerGradCam(model, last_conv).attribute(inp, target=0)
    gc = LayerAttribution.interpolate(gc, (H, W)).detach().relu().mean(dim=(0, 1)).numpy()
    out["grad_cam"] = float(np.corrcoef(gc.ravel(), corr.ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    t0 = time.time()
    import pandas as pd
    pred_rows, attr_rows = [], []

    GATES = {"ungated": None,
             "ours_TVmask": lambda H, W: MaskGate(H, W, 1),
             "L0_gate": lambda H, W: L0Gate(H, W),
             "STG_gate": lambda H, W: STGGate(H, W),
             "attention": lambda H, W: SpatialAttnGate(H, W)}

    for lead in args.leads:
        X, y, a, b = splits(lead)
        H, W = X.shape[2], X.shape[3]
        corr = np.abs(ref[f"lead{lead}_corr"])
        Xf, yv = X.reshape(len(y), -1), y
        sc = StandardScaler().fit(Xf[:b])
        Xtr_s, Xte_s = sc.transform(Xf[:b]), sc.transform(Xf[b:])

        # --- linear + tree predictive baselines ---
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_s, yv[:b])
        pred_rows.append({"lead": lead, "method": "logistic_L2", "type": "linear",
                          "test_auroc": roc_auc_score(yv[b:], lr.predict_proba(Xte_s)[:, 1]), "auroc_sd": 0.0})
        coefmap = np.abs(lr.coef_.reshape(H, W))
        attr_rows.append({"lead": lead, "method": "logistic_L2", "agree": float(np.corrcoef(coefmap.ravel(), corr.ravel())[0, 1])})

        aug = [roc_auc_score(yv[b:], HistGradientBoostingClassifier(random_state=s, max_iter=200)
                             .fit(Xf[:b], yv[:b]).predict_proba(Xf[b:])[:, 1]) for s in range(args.seeds)]
        pred_rows.append({"lead": lead, "method": "grad_boosting", "type": "nonlinear",
                          "test_auroc": np.mean(aug), "auroc_sd": np.std(aug)})

        # --- CNN-based methods ---
        ungated_models = []
        for mname, mk in GATES.items():
            aus, agrs, imaps = [], [], []
            for s in range(args.seeds):
                r = fit_cnn(mk, lead, s, args.epochs)
                aus.append(r["test_auroc"])
                if mname == "ungated":
                    ungated_models.append((r["model"], X[b:]))
                if r["imap"] is not None:
                    imaps.append(r["imap"])
            pred_rows.append({"lead": lead, "method": ("CNN_" + mname), "type": "nonlinear",
                              "test_auroc": np.mean(aus), "auroc_sd": np.std(aus)})
            if imaps:
                avg = np.mean(imaps, 0)
                attr_rows.append({"lead": lead, "method": mname, "agree": float(np.corrcoef(avg.ravel(), corr.ravel())[0, 1])})

        # --- post-hoc attributions on one ungated model ---
        m0, Xte0 = ungated_models[0]
        for k, v in posthoc_agreements(m0, Xte0, corr, H, W).items():
            attr_rows.append({"lead": lead, "method": "posthoc_" + k, "agree": v})
        print(f"  lead {lead} done")

    pred = pd.DataFrame(pred_rows); attr = pd.DataFrame(attr_rows)
    pred.to_csv(REPO_ROOT / "results" / "sst_methods_predictive.csv", index=False)
    attr.to_csv(REPO_ROOT / "results" / "sst_methods_attribution.csv", index=False)

    for lead in args.leads:
        print(f"\n=== Lead {lead}: PREDICTIVE skill (test AUROC) ===")
        p = pred[pred.lead == lead].sort_values("test_auroc", ascending=False)
        for _, r in p.iterrows():
            print(f"  {r['method']:<16} [{r['type']:<9}]  AUROC={r['test_auroc']:.3f} ± {r['auroc_sd']:.3f}")
        print(f"--- Lead {lead}: ATTRIBUTION agreement with correlation reference ---")
        ag = attr[attr.lead == lead].sort_values("agree", ascending=False)
        for _, r in ag.iterrows():
            print(f"  {r['method']:<18} agree={r['agree']:+.3f}")
    print(f"\nTotal {time.time()-t0:.0f}s. Wrote results/sst_methods_predictive.csv, sst_methods_attribution.csv")


if __name__ == "__main__":
    main()
