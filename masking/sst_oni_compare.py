"""Rigorous ENSO labels + a small tuning sweep, to make sense of the linear-vs-CNN picture.

Fixes the soft spots in the original label:
  - ONI-style target: trailing 3-month running mean of the Nino-3.4 anomaly (causal, no leakage),
    thresholded at 0.5 C.
  - TRAIN-ONLY climatology: monthly means fit on the training block only.
  - linear DETRENDING (train-fit, per cell + Nino series): removes the warming-trend confound so
    no model can ride the long-term trend instead of ENSO dynamics.

Then compares, on the same 3-way temporal split, on BOTH OISST (43 yr) and ERSST (172 yr):
  persistence (current ONI) | logistic (full field) | a few tuned flatten-CNNs (best wins).

Usage:  .venv/bin/python masking/sst_oni_compare.py --seeds 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import INPUT_LAT, INPUT_LON, NINO34_LAT, NINO34_LON  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cpu")
DATASETS = {"OISST(43yr)": ("data/sst_data.nc", 4), "ERSST(172yr)": ("data/sst.mnmean.nc", 1)}


def _detrend(A, ntr):                      # A (T,...) ; remove train-fit linear trend along time
    T = A.shape[0]; t = np.arange(T, dtype=float); tr = t[:ntr]; trm = tr.mean()
    Atr = A[:ntr]; denom = ((tr - trm) ** 2).sum()
    sl = (((tr - trm).reshape((-1,) + (1,) * (A.ndim - 1))) * (Atr - Atr.mean(0))).sum(0) / denom
    ic = Atr.mean(0) - sl * trm
    return A - (sl[None] * t.reshape((-1,) + (1,) * (A.ndim - 1)) + ic[None])


def _trailing(v, k):                       # causal k-month running mean (first k-1 = nan)
    c = np.cumsum(np.insert(v, 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def load_oni(lead, file_path, resolution, train_frac=0.8, smooth=3):
    sst = xr.open_dataset(REPO_ROOT / file_path)["sst"].sortby("lat", ascending=False).sortby("lon")
    broad = sst.sel(lat=INPUT_LAT, lon=INPUT_LON)
    nino = broad.sel(lat=NINO34_LAT, lon=NINO34_LON).mean(["lat", "lon"])
    T = broad.sizes["time"]; ntr = int(train_frac * T)
    clim = broad.isel(time=slice(0, ntr)).groupby("time.month").mean("time")
    anom = (broad.groupby("time.month") - clim).values                       # (T,Hf,Wf)
    nclim = nino.isel(time=slice(0, ntr)).groupby("time.month").mean("time")
    nanom = (nino.groupby("time.month") - nclim).values                      # (T,)
    anom = _detrend(anom, ntr); nanom = _detrend(nanom, ntr)
    oni = _trailing(nanom, smooth)                                           # causal ONI
    Xfield = anom[:, ::resolution, ::resolution][:, None].astype(np.float32)  # (T,1,H,W)
    # align: drop first smooth-1 (nan ONI); label = ONI at t+lead > 0.5
    s = smooth - 1
    y = (oni[s + lead:] > 0.5).astype(np.float32)
    X = Xfield[s:s + len(y)]
    cur = oni[s:s + len(y)].reshape(-1, 1)                                   # current ONI (persistence)
    return X, y, cur


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


def fit_cnn(X, y, a, b, seed, channels, bs, wd, epochs=120):
    Xtr = torch.tensor(X[:a]); ytr = torch.tensor(y[:a]).view(-1, 1)
    Xva, yva, Xte, yte = X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = FlatCNN(channels, X.shape[2], X.shape[3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd); bce = nn.BCEWithLogitsLoss()
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


CNN_CFGS = {"cnn[16,32]bs16": ([16, 32], 16, 0.0),
            "cnn[16,32]wd1e-4": ([16, 32], 16, 1e-4),
            "cnn[16,32,64]": ([16, 32, 64], 16, 0.0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=2); args = ap.parse_args()
    t0 = time.time()
    for dname, (path, res) in DATASETS.items():
        for lead in [3, 6]:
            X, y, cur = load_oni(lead, path, res)
            T = len(y); a, b = int(0.6 * T), int(0.8 * T)
            rate = y.mean()
            # persistence (current ONI, 1 feature)
            lp = LogisticRegression(max_iter=1000).fit(cur[:b], y[:b])
            ap_ = roc_auc_score(y[b:], lp.predict_proba(cur[b:])[:, 1])
            # logistic full field
            Xf = X.reshape(T, -1); sc = StandardScaler().fit(Xf[:b])
            lf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xf[:b]), y[:b])
            af = roc_auc_score(y[b:], lf.predict_proba(sc.transform(Xf[b:]))[:, 1])
            print(f"\n[{dname} lead {lead}] grid {X.shape[2]}x{X.shape[3]}, N={T}, event-rate={rate:.2f}")
            print(f"   persistence(curr ONI) = {ap_:.3f}   logistic(field) = {af:.3f}")
            best = (-1, "")
            for cname, (ch, bs, wd) in CNN_CFGS.items():
                cs = [fit_cnn(X, y, a, b, s, ch, bs, wd) for s in range(args.seeds)]
                m = np.mean(cs)
                print(f"   {cname:<18} = {m:.3f} ± {np.std(cs):.3f}")
                if m > best[0]:
                    best = (m, cname)
            print(f"   --> best CNN = {best[1]} ({best[0]:.3f});  logistic {'WINS' if af>=best[0] else 'loses'} by {af-best[0]:+.3f}")
    print(f"\nTotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
