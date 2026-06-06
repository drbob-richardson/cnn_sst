"""WeatherBench first test: is this regime genuinely nonlinear (CNN > linear)?

Task (analogous to ENSO, but data-rich and dynamically nonlinear): from the GLOBAL Z500 anomaly
field at time t, predict whether a mid-latitude TARGET region's Z500 anomaly is in its top tercile
`lead` days later. Deseasonalized with a train-only monthly climatology; strict temporal 3-way
split. Compare persistence (current target value) / logistic (global field) / CNN (flatten head).

If the CNN beats logistic here, we have the nonlinear, data-rich showcase the method needs (and
the mask should be identifiable). Runs on the local cache from wb_pull.py.

Usage:  .venv/bin/python masking/wb_compare.py --leads-days 2 5 --seeds 2
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

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "results" / "wb_z500.npz"
# target region: eastern North Pacific / NW North America storm track (active mid-latitude variability)
TGT_LAT = (40, 58)
TGT_LON = (200, 245)


class FlatCNN(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        chs = [1] + list(channels); layers = []
        for i in range(len(chs) - 1):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1), nn.BatchNorm2d(chs[i + 1]), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            fd = self.features(torch.zeros(1, 1, H, W)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fd, 1))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def fit_cnn(X, y, a, b, seed, channels=(16, 32), bs=64, epochs=40):
    Xtr = torch.tensor(X[:a]); ytr = torch.tensor(y[:a]).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = FlatCNN(channels, X.shape[2], X.shape[3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    best_va, best_te = -1, float("nan")
    for ep in range(epochs):
        model.train()
        for i in torch.randperm(a).split(bs):
            loss = bce(model(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva))).numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best_va:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte))).numpy().ravel()
            best_va, best_te = va, roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan")
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads-days", type=int, nargs="+", default=[2, 5])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    d = np.load(CACHE)
    z, month, lat, lon = d["z"], d["month"], d["lat"], d["lon"]
    T = len(z); ntr = int(0.8 * T)
    # train-only monthly climatology -> anomalies
    clim = np.stack([z[:ntr][month[:ntr] == m].mean(0) for m in range(1, 13)])  # (12,H,W)
    anom = (z - clim[month - 1]).astype(np.float32)
    # target-region mean anomaly series
    li = np.where((lat >= TGT_LAT[0]) & (lat <= TGT_LAT[1]))[0]
    lj = np.where((lon >= TGT_LON[0]) & (lon <= TGT_LON[1]))[0]
    tgt = anom[:, li][:, :, lj].mean(axis=(1, 2))                                 # (T,)
    print(f"data: {T} 6-hourly steps ({T//1460} yr), grid {z.shape[1]}x{z.shape[2]}, "
          f"target box {len(li)}x{len(lj)} cells")

    def evaluate(X, cur, y, tag, nseeds):
        N = len(y); a, b = int(0.6 * N), int(0.8 * N)
        sc = StandardScaler().fit(cur[:b]); lp = LogisticRegression(max_iter=1000).fit(sc.transform(cur[:b]), y[:b])
        ap_ = roc_auc_score(y[b:], lp.predict_proba(sc.transform(cur[b:]))[:, 1])
        Xf = X.reshape(N, -1); scf = StandardScaler().fit(Xf[:b])
        lf = LogisticRegression(max_iter=2000, C=1.0).fit(scf.transform(Xf[:b]), y[:b])
        af = roc_auc_score(y[b:], lf.predict_proba(scf.transform(Xf[b:]))[:, 1])
        cs = [fit_cnn(X, y, a, b, s) for s in range(nseeds)]
        cm = np.mean(cs)
        print(f"  {tag:<18} N={N} rate={y.mean():.2f}  persist={ap_:.3f}  logistic={af:.3f}  "
              f"CNN={cm:.3f}±{np.std(cs):.3f}  -> CNN {'BEATS' if cm > af else 'loses'} by {cm-af:+.3f}")

    for ld in args.leads_days:
        k = ld * 4
        ntr_y = int(0.8 * (len(tgt) - k))
        X = anom[:len(tgt) - k, None]
        cur = tgt[:len(tgt) - k].reshape(-1, 1)
        print(f"\n[lead {ld}d]")
        # (1) LEVEL target: top-tercile absolute Z500 anomaly at t+lead
        lvl = tgt[k:]
        evaluate(X, cur, (lvl > np.quantile(lvl[:ntr_y], 2 / 3)).astype(np.float32), "level (abs)", args.seeds)
        # (2) TENDENCY target: top-tercile CHANGE over the lead (removes persistence)
        dy = tgt[k:] - tgt[:len(tgt) - k]
        evaluate(X, cur, (dy > np.quantile(dy[:ntr_y], 2 / 3)).astype(np.float32), "tendency (change)", args.seeds)
    print(f"\ndone")


if __name__ == "__main__":
    main()
