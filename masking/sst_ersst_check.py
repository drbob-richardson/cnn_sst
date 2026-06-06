"""Does 4x more data (ERSST, 1854-2026) change the linear-vs-CNN verdict on ENSO?

Our learning curve (within the 43-yr OISST record) predicted more data favors the LINEAR model.
ERSST gives ~172 years (~2069 months, ~4x), a real test rather than extrapolation. Same 3-way
temporal split, same flatten CNN; compare test AUROC of logistic vs CNN.

Usage:  .venv/bin/python masking/sst_ersst_check.py --data data/sst.mnmean.nc --resolution 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_final import FinalCNN, BS, EPOCHS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def cnn_auroc(X, y, a, b, seed):
    Xtr = torch.tensor(X[:a], dtype=torch.float32); ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = FinalCNN(H=X.shape[2], W=X.shape[3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    best_va, best_te = -1, float("nan")
    for ep in range(EPOCHS):
        model.train()
        for i in torch.randperm(a).split(BS):
            loss = bce(model(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best_va:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
            best_va, best_te = va, roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan")
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/sst.mnmean.nc")
    ap.add_argument("--resolution", type=int, default=1)
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    path = REPO_ROOT / args.data
    t0 = time.time()
    for lead in args.leads:
        X, y = process_data_multi_res(lead, resolution=args.resolution, file_path=path)
        T = len(y); a, b = int(0.6 * T), int(0.8 * T)
        runs = int(np.sum((y[1:] == 1) & (y[:-1] == 0)) + (1 if y[0] == 1 else 0))
        Xf = X.reshape(T, -1); sc = StandardScaler().fit(Xf[:b])
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xf[:b]), y[:b])
        la = roc_auc_score(y[b:], lr.predict_proba(sc.transform(Xf[b:]))[:, 1])
        cs = [cnn_auroc(X, y, a, b, s) for s in range(args.seeds)]
        print(f"  lead {lead}: grid={X.shape[2]}x{X.shape[3]}, N={T} months, ~{runs} events, "
              f"train={a} | logistic={la:.3f}  CNN(flatten)={np.mean(cs):.3f}±{np.std(cs):.3f}  gap={la-np.mean(cs):+.3f}")
    print(f"\nDevice cpu. Total {time.time()-t0:.0f}s  (data: {args.data}, resolution {args.resolution})")


if __name__ == "__main__":
    main()
