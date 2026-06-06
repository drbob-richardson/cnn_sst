"""Can the CNN be redeemed on SST? Test location-aware backbones vs. the GAP CNN and logistic.

Diagnosis: the GAP CNN discards absolute location, but ENSO prediction is a location-specific
question (is the anomaly in the Nino box?), which a per-cell logistic captures directly. Here we
compare, on the same 3-way temporal split (train 60% / val 20% early-stop / test 20%):

  - logistic (linear, location-aware by construction)         [reference ceiling]
  - GAP CNN          (current; global average pooling)        [location-destroying]
  - Flatten CNN      (conv features flattened, not pooled)    [location-aware head]
  - CoordConv + GAP  (coordinate channels added to the input) [location-aware conv]
  - CoordConv + Flatten

Prediction only (ungated). Usage:  .venv/bin/python masking/sst_arch_explore.py --leads 3 6
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "sst_data.nc"
DEVICE = torch.device("cpu")


class ArchCNN(nn.Module):
    def __init__(self, channels=(16, 32), pool=True, head="gap", coord=False, H=30, W=90):
        super().__init__()
        self.coord = coord
        in0 = 1 + (2 if coord else 0)
        chs = [in0] + list(channels)
        layers = []
        for i in range(len(chs) - 1):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU()]
            if pool:
                layers += [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head_mode = head
        if head == "gap":
            self.head = nn.Linear(channels[-1], 1)
        else:
            with torch.no_grad():
                fd = self._feat(torch.zeros(1, 1, H, W)).flatten(1).shape[1]
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fd, 1))

    def _feat(self, x):
        if self.coord:
            B, _, H, W = x.shape
            yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([x, yy, xx], 1)
        return self.features(x)

    def forward(self, x):
        f = self._feat(x)
        f = f.mean(dim=(2, 3)) if self.head_mode == "gap" else f.flatten(1)
        return self.head(f)


ARCHS = {
    "gap":            dict(head="gap", coord=False),
    "flatten":        dict(head="flatten", coord=False),
    "coord_gap":      dict(head="gap", coord=True),
    "coord_flatten":  dict(head="flatten", coord=True),
}


def fit(arch_kw, lead, seed, epochs=60):
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    T = len(y); a, b = int(0.6 * T), int(0.8 * T)
    Xtr = torch.tensor(X[:a], dtype=torch.float32); ytr = torch.tensor(y[:a], dtype=torch.float32).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = ArchCNN(**arch_kw).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    n, bs = a, 128
    best_va, best_te = -1, float("nan")
    for ep in range(epochs):
        model.train()
        for i in torch.randperm(n).split(bs):
            xb = Xtr[i].to(DEVICE)
            xb = xb * (torch.rand_like(xb) > 0.20) / 0.80
            loss = bce(model(xb), ytr[i].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best_va:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
            best_va, best_te = va, roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan")
    return best_te, sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    import pandas as pd
    rows = []
    for lead in args.leads:
        X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
        T = len(y); b = int(0.8 * T)
        Xf = X.reshape(T, -1)
        sc = StandardScaler().fit(Xf[:b])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf[:b]), y[:b])
        rows.append({"lead": lead, "arch": "logistic (linear)", "params": Xf.shape[1] + 1,
                     "test_auroc": roc_auc_score(y[b:], lr.predict_proba(sc.transform(Xf[b:]))[:, 1]), "sd": 0.0})
        for name, kw in ARCHS.items():
            res = [fit(kw, lead, s, args.epochs) for s in range(args.seeds)]
            aus = [r[0] for r in res]
            rows.append({"lead": lead, "arch": name, "params": res[0][1],
                         "test_auroc": np.mean(aus), "sd": np.std(aus)})
        print(f"  lead {lead} done")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_arch_explore.csv", index=False)
    for lead in args.leads:
        print(f"\n=== Lead {lead}: test AUROC by architecture ===")
        for _, r in df[df.lead == lead].sort_values("test_auroc", ascending=False).iterrows():
            print(f"  {r['arch']:<18} params={int(r['params']):>7}  AUROC={r['test_auroc']:.3f} ± {r['sd']:.3f}")
    print(f"\nTotal {time.time()-t0:.0f}s -> results/sst_arch_explore.csv")


if __name__ == "__main__":
    main()
