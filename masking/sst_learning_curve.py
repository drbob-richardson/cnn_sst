"""Is the CNN losing to linear because the training set is too small?

Learning curve: train logistic and the (flatten) CNN on increasing fractions of the train block,
evaluate on the fixed test block. If the CNN-vs-logistic gap SHRINKS as data grows, the CNN is
data-limited (more data would help, matching the ENSO-DL literature's reliance on climate-model
pretraining). If the gap is flat, the relationship is genuinely linear at this resolution.

Usage:  .venv/bin/python masking/sst_learning_curve.py --leads 3 6 --seeds 3
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_final import FinalCNN, BS, EPOCHS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "sst_data.nc"
FRACS = [0.25, 0.5, 0.75, 1.0]


def cnn_auroc(X, y, ntr_full, ntr_keep, b, seed):
    s0 = ntr_full - ntr_keep                       # keep the most-recent ntr_keep months of train
    Xtr = torch.tensor(X[s0:ntr_full], dtype=torch.float32); ytr = torch.tensor(y[s0:ntr_full], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[ntr_full:b], y[ntr_full:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = FinalCNN(H=X.shape[2], W=X.shape[3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    n = len(Xtr); best_va, best_te = -1, float("nan")
    for ep in range(EPOCHS):
        model.train()
        for i in torch.randperm(n).split(BS):
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
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import pandas as pd
    rows = []; t0 = time.time()
    fig, axes = plt.subplots(1, len(args.leads), figsize=(5.2 * len(args.leads), 4.2), squeeze=False)
    for c, lead in enumerate(args.leads):
        X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
        T = len(y); ntr_full = int(0.6 * T); b = int(0.8 * T)
        Xf = X.reshape(T, -1)
        for frac in FRACS:
            nk = int(frac * ntr_full); s0 = ntr_full - nk
            sc = StandardScaler().fit(Xf[s0:ntr_full])
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf[s0:ntr_full]), y[s0:ntr_full])
            la = roc_auc_score(y[b:], lr.predict_proba(sc.transform(Xf[b:]))[:, 1])
            cs = [cnn_auroc(X, y, ntr_full, nk, b, s) for s in range(args.seeds)]
            rows.append({"lead": lead, "frac": frac, "n_train": nk, "logistic": la,
                         "cnn": np.mean(cs), "cnn_sd": np.std(cs), "gap": la - np.mean(cs)})
            print(f"  lead {lead} n_train={nk:<4} logistic={la:.3f}  CNN={np.mean(cs):.3f}±{np.std(cs):.3f}  gap={la-np.mean(cs):+.3f}")
        sub = pd.DataFrame([r for r in rows if r["lead"] == lead])
        ax = axes[0, c]
        ax.plot(sub.n_train, sub.logistic, "o-", label="logistic (linear)", color="#2c7fb8")
        ax.errorbar(sub.n_train, sub.cnn, yerr=sub.cnn_sd, fmt="s-", label="CNN (flatten)", color="#d95f0e", capsize=3)
        ax.set_title(f"Lead {lead}"); ax.set_xlabel("# training months"); ax.set_ylabel("test AUROC"); ax.legend(fontsize=8)
    fig.suptitle("Learning curves: does more data close the CNN-vs-linear gap?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(REPO_ROOT / "write_up" / "sst_learning_curve.png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(REPO_ROOT / "results" / "sst_learning_curve.csv", index=False)
    print(f"\nTotal {time.time()-t0:.0f}s -> results/sst_learning_curve.csv, write_up/sst_learning_curve.png")


if __name__ == "__main__":
    main()
