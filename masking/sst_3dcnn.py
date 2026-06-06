"""Does TEMPORAL context redeem the CNN? A 3D CNN over 3 years of prior SST.

The 2D single-snapshot task was dominated by linear persistence. ENSO's nonlinearity lives in the
temporal evolution, so here each sample is the SST-anomaly field for the previous `window` months
(default 36 = 3 years), fed to a 3D CNN. We compare, on the same 3-way temporal split:

  - linear AR : logistic on the past `window` Nino-3.4 index values (the standard linear baseline)
  - linear full: logistic on the flattened space-time cube (strong L2)
  - 3D CNN    : Conv3d over (time, lat, lon) with a location-preserving head

If the 3D CNN beats the linear baselines, the temporal nonlinearity is real and exploitable
(worth pursuing the masked version + more data). If not, even temporal framing is near-linear here.

Usage:  .venv/bin/python masking/sst_3dcnn.py --leads 3 6 --window 36 --seeds 3
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
DEVICE = torch.device("cpu")   # 3D max-pool isn't implemented on MPS; CPU + strided convs is fine here
# Nino-3.4 box on the 30x90 grid (lat ~[-5,5] rows 10:20, lon ~[190,240] cols 20:70)
NINO = (slice(10, 20), slice(20, 70))


def build_windows(lead, window, resolution=4):
    X, y = process_data_multi_res(lead, resolution=resolution, file_path=DATA)
    X = X[:, 0]                                          # (T, H, W)
    T = len(y)
    cubes, labels, nino_hist = [], [], []
    for t in range(window - 1, T):
        cubes.append(X[t - window + 1:t + 1])           # (window, H, W)
        labels.append(y[t])
        nino_hist.append(X[t - window + 1:t + 1, NINO[0], NINO[1]].mean(axis=(1, 2)))
    return np.array(cubes, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(nino_hist, dtype=np.float32)


class CNN3D(nn.Module):
    def __init__(self, window, H, W):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1), nn.BatchNorm3d(8), nn.ReLU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 8)),            # keep coarse space-time location
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(16 * 4 * 4 * 8, 1))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def fit_cnn3d(cubes, labels, a, b, seed, epochs=80, bs=16):
    Xtr = torch.tensor(cubes[:a]).unsqueeze(1); ytr = torch.tensor(labels[:a]).view(-1, 1)
    Xva, yva = cubes[a:b], labels[a:b]; Xte, yte = cubes[b:], labels[b:]
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN3D(*cubes.shape[1:]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); bce = nn.BCEWithLogitsLoss()
    n = a; best_va, best_te = -1, float("nan")
    for ep in range(epochs):
        model.train()
        for i in torch.randperm(n).split(bs):
            xb = Xtr[i].to(DEVICE)
            loss = bce(model(xb), ytr[i].to(DEVICE)); opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(torch.tensor(Xva).unsqueeze(1).to(DEVICE))).cpu().numpy().ravel()
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best_va:
            with torch.no_grad():
                pt = torch.sigmoid(model(torch.tensor(Xte).unsqueeze(1).to(DEVICE))).cpu().numpy().ravel()
            best_va, best_te = va, roc_auc_score(yte, pt) if len(np.unique(yte)) > 1 else float("nan")
    return best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--window", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import pandas as pd
    rows = []; t0 = time.time()
    for lead in args.leads:
        cubes, y, ninoh = build_windows(lead, args.window)
        N = len(y); a, b = int(0.6 * N), int(0.8 * N)
        # linear AR on Nino-3.4 history
        sc = StandardScaler().fit(ninoh[:b])
        lr_ar = LogisticRegression(max_iter=2000).fit(sc.transform(ninoh[:b]), y[:b])
        ar = roc_auc_score(y[b:], lr_ar.predict_proba(sc.transform(ninoh[b:]))[:, 1])
        # linear full on flattened cube
        Cf = cubes.reshape(N, -1); scf = StandardScaler().fit(Cf[:b])
        lr_f = LogisticRegression(max_iter=2000, C=0.1).fit(scf.transform(Cf[:b]), y[:b])
        full = roc_auc_score(y[b:], lr_f.predict_proba(scf.transform(Cf[b:]))[:, 1])
        # 3D CNN
        cs = [fit_cnn3d(cubes, y, a, b, s) for s in range(args.seeds)]
        rows.append({"lead": lead, "n_samples": N, "linear_AR": ar, "linear_full": full,
                     "cnn3d": np.mean(cs), "cnn3d_sd": np.std(cs)})
        print(f"  lead {lead}: N={N}  linear_AR={ar:.3f}  linear_full={full:.3f}  CNN3D={np.mean(cs):.3f}±{np.std(cs):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_3dcnn.csv", index=False)
    print(f"\n=== Does the 3D CNN (3-yr history) beat the linear baselines? ===")
    print(df.to_string(index=False))
    print(f"\nDevice {DEVICE}. Total {time.time()-t0:.0f}s -> results/sst_3dcnn.csv")


if __name__ == "__main__":
    main()
