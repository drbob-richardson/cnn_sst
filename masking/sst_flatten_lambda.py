"""Re-tune the mask penalties on the redeemed (flatten) backbone.

On the location-preserving flatten CNN the pixel mask lost agreement with the correlation
reference (~0) at the GAP-tuned weight lambda_sp=1e-2. Hypothesis: a stronger sparsity penalty
forces the (compensating) head to keep the predictive equatorial pixels, recovering an
interpretable mask. We sweep lambda_sp (x two lambda_tv) and report, on the test block, both the
mask<->reference agreement AND test AUROC -- a recovery only counts if AUROC stays competitive.

Usage:  .venv/bin/python masking/sst_flatten_lambda.py --leads 3 6 --seeds 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_final import FinalCNN, BS, EPOCHS  # noqa: E402
from masking.sst_masking import MaskGate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "sst_data.nc"


def fit(lead, seed, lam_sp, lam_tv, tile=1):
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    H, W = X.shape[2], X.shape[3]; T = len(y); a, b = int(0.6 * T), int(0.8 * T)
    Xtr = torch.tensor(X[:a], dtype=torch.float32); ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    gate = MaskGate(H, W, tile)
    model = FinalCNN(gate=gate, H=H, W=W)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    best = {"va": -1, "auroc": float("nan"), "imap": None, "mean": float("nan")}
    for ep in range(EPOCHS):
        gate.tau = 2.0 - 1.99 * (ep / (EPOCHS - 1))
        model.train()
        for i in torch.randperm(a).split(BS):
            loss = bce(model(Xtr[i]), ytr[i]) + gate.penalty(lam_sp, lam_tv)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best["va"]:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
            m = gate.importance()
            best = {"va": va, "auroc": roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan"),
                    "imap": m, "mean": float(m.mean())}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    SP = [0.01, 0.05, 0.1, 0.3, 1.0]
    TV = [0.05, 0.2]
    import pandas as pd
    rows = []
    t0 = time.time()
    for lead in args.leads:
        corr = np.abs(ref[f"lead{lead}_corr"])
        for ltv in TV:
            for lsp in SP:
                au, ag, mn = [], [], []
                for s in range(args.seeds):
                    r = fit(lead, s, lsp, ltv)
                    au.append(r["auroc"]); mn.append(r["mean"])
                    ag.append(float(np.corrcoef(r["imap"].ravel(), corr.ravel())[0, 1]))
                rows.append({"lead": lead, "lam_sp": lsp, "lam_tv": ltv,
                             "auroc": np.mean(au), "agree": np.mean(ag), "mask_mean": np.mean(mn)})
                print(f"  lead {lead} lam_sp={lsp:<5} lam_tv={ltv:<4} AUROC={np.mean(au):.3f} "
                      f"agree={np.mean(ag):+.3f} mask_mean={np.mean(mn):.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_flatten_lambda.csv", index=False)
    print(f"\n=== summary: did agreement recover at competitive AUROC? ===")
    for lead in args.leads:
        sub = df[df.lead == lead].sort_values("agree", ascending=False)
        print(f"\nLead {lead} (logistic ceiling ~{0.924 if lead==3 else 0.793}):")
        for _, r in sub.iterrows():
            print(f"  lam_sp={r['lam_sp']:<5} lam_tv={r['lam_tv']:<4}  agree={r['agree']:+.3f}  AUROC={r['auroc']:.3f}  mask_mean={r['mask_mean']:.2f}")
    print(f"\nTotal {time.time()-t0:.0f}s -> results/sst_flatten_lambda.csv")


if __name__ == "__main__":
    main()
