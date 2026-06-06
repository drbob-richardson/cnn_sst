"""One-factor-at-a-time ablation of CNN design choices for SST prediction.

Motivated by two clues: (a) global average pooling discards location, which the per-cell logistic
keeps; (b) batch 128 on ~307 training months is only ~3 gradient updates/epoch (severe
under-optimization). We vary one factor at a time from a sensible base and ask which choices close
the gap to the logistic ceiling. Same 3-way temporal split; prediction only (ungated); test AUROC.

Usage:  .venv/bin/python masking/sst_cnn_ablation.py --leads 3 6 --seeds 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "sst_data.nc"
DEVICE = torch.device("cpu")


class CNN(nn.Module):
    def __init__(self, channels, pool, head, H=30, W=90):
        super().__init__()
        chs = [1] + list(channels)
        layers = []
        for i in range(len(chs) - 1):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU()]
            if pool == "max":
                layers += [nn.MaxPool2d(2)]
        if pool == "adapt4":
            layers += [nn.AdaptiveAvgPool2d(4)]
        self.features = nn.Sequential(*layers)
        self.head_mode = head
        if head == "gap":
            self.head = nn.Linear(channels[-1], 1)
        else:
            with torch.no_grad():
                fd = self.features(torch.zeros(1, 1, H, W)).flatten(1).shape[1]
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fd, 1))

    def forward(self, x):
        f = self.features(x)
        f = f.mean(dim=(2, 3)) if self.head_mode == "gap" else f.flatten(1)
        return self.head(f)


BASE = dict(channels=[16, 32], pool="max", head="flatten", bs=32, epochs=120, lr=1e-3, drop=0.0, wd=0.0)
CONFIGS = {
    "BASE (flatten, bs32)": BASE,
    "batch=128 (3 upd/ep)": {**BASE, "bs": 128},
    "batch=16":             {**BASE, "bs": 16},
    "batch=8":              {**BASE, "bs": 8},
    "+ input dropout 0.2":  {**BASE, "drop": 0.2},
    "head = GAP":           {**BASE, "head": "gap"},
    "1 conv layer":         {**BASE, "channels": [16]},
    "3 conv layers":        {**BASE, "channels": [16, 32, 64]},
    "adaptive-pool 4x4":    {**BASE, "pool": "adapt4"},
    "no pooling (full)":    {**BASE, "pool": "none"},
    "+ weight decay 1e-3":  {**BASE, "wd": 1e-3},
    "tuned (bs16,wd,long)": {**BASE, "bs": 16, "wd": 1e-4, "epochs": 200},
}


def fit(cfg, lead, seed):
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    T = len(y); a, b = int(0.6 * T), int(0.8 * T)
    Xtr = torch.tensor(X[:a], dtype=torch.float32); ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN(cfg["channels"], cfg["pool"], cfg["head"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    bce = nn.BCEWithLogitsLoss()
    n, bs = a, cfg["bs"]
    best_va, best_te = -1, float("nan")
    for ep in range(cfg["epochs"]):
        model.train()
        for i in torch.randperm(n).split(bs):
            xb = Xtr[i]
            if cfg["drop"] > 0:
                xb = xb * (torch.rand_like(xb) > cfg["drop"]) / (1 - cfg["drop"])
            loss = bce(model(xb), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
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
    t0 = time.time()
    import pandas as pd
    rows = []
    for lead in args.leads:
        X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
        T = len(y); b = int(0.8 * T); Xf = X.reshape(T, -1)
        sc = StandardScaler().fit(Xf[:b])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf[:b]), y[:b])
        rows.append({"lead": lead, "config": "logistic (ceiling)",
                     "auroc": roc_auc_score(y[b:], lr.predict_proba(sc.transform(Xf[b:]))[:, 1]), "sd": 0.0})
        hgb = [roc_auc_score(y[b:], HistGradientBoostingClassifier(random_state=s, max_iter=200)
                             .fit(Xf[:b], y[:b]).predict_proba(Xf[b:])[:, 1]) for s in range(args.seeds)]
        rows.append({"lead": lead, "config": "gradient boosting", "auroc": np.mean(hgb), "sd": np.std(hgb)})
        for name, cfg in CONFIGS.items():
            aus = [fit(cfg, lead, s) for s in range(args.seeds)]
            rows.append({"lead": lead, "config": name, "auroc": np.mean(aus), "sd": np.std(aus)})
            print(f"  lead {lead} {name:<22} AUROC={np.mean(aus):.3f} ± {np.std(aus):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_cnn_ablation.csv", index=False)
    for lead in args.leads:
        print(f"\n=== Lead {lead}: test AUROC (sorted) ===")
        for _, r in df[df.lead == lead].sort_values("auroc", ascending=False).iterrows():
            print(f"  {r['config']:<22} {r['auroc']:.3f} ± {r['sd']:.3f}")
    print(f"\nTotal {time.time()-t0:.0f}s -> results/sst_cnn_ablation.csv")


if __name__ == "__main__":
    main()
