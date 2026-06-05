"""Reconstructed El Nino input-masking driver (replaces the lost learn_pixel_masks_loop_serial.py).

Backbone follows the paper's spec: a DeepCNN (five 3x3 convs 1->32->64->64->128->128, each with
BatchNorm + ReLU), global average pooling, MLP head (128->64->32->1, dropout 0.3). Three input
gates --- pixel-wise, 2x2 tiles, 4x4 tiles --- are trained jointly with sparsity + total-variation
penalties and temperature annealing (tau 2.0 -> 0.01); 'none' is the ungated baseline.

Runs on the 30x90 downsampled SST field (resolution=4) so masks are directly comparable to the
lagged-correlation statistical reference (results/sst_statistical_reference.npz). For each lead and
gate we report accuracy / F1 / AUROC over several seeds, save the seed-averaged mask maps, and
quantify agreement between each mask and the correlation reference.

Usage:  .venv/bin/python masking/sst_masking.py --leads 3 6 --seeds 4 --epochs 60
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
DATA = REPO_ROOT / "data" / "sst_data.nc"
EXTENT = [170, 260, -15, 15]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MaskGate(nn.Module):
    """Learnable input mask shared within tile x tile blocks (tile=1 -> pixel-wise)."""
    def __init__(self, H, W, tile=1):
        super().__init__()
        self.H, self.W, self.tile = H, W, tile
        self.nh, self.nw = math.ceil(H / tile), math.ceil(W / tile)
        self.z = nn.Parameter(torch.zeros(self.nh, self.nw))
        self.tau = 1.0

    def small(self):
        return torch.sigmoid(self.z / self.tau)

    def map(self):
        m = self.small()
        m = m.repeat_interleave(self.tile, 0).repeat_interleave(self.tile, 1)
        return m[: self.H, : self.W]

    def forward(self, x):
        m = self.map().to(x.device)
        return x * m.view(1, 1, self.H, self.W), m

    def penalty(self, lam_sp, lam_tv):
        m = self.small()
        tv = (m[:, 1:] - m[:, :-1]).abs().sum() + (m[1:, :] - m[:-1, :]).abs().sum()
        return lam_sp * (m ** 2).mean() + lam_tv * tv / m.numel()

    def importance(self):
        return self.map().detach().cpu().numpy()


class DeepCNN(nn.Module):
    def __init__(self, gate=None):
        super().__init__()
        self.gate = gate
        ch = [1, 32, 64, 64, 128, 128]
        layers = []
        for i in range(5):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if self.gate is not None:
            x, _ = self.gate(x)
        x = self.features(x).mean(dim=(2, 3))      # global average pool
        return self.head(x)


def train_eval(lead, tile, seed, epochs, resolution=4, lam_sp=1e-2, lam_tv=1e-2):
    """tile=None -> ungated baseline; tile>=1 -> masked. Returns (metrics, mask_map_or_None)."""
    X, y = process_data_multi_res(lead, resolution=resolution, file_path=DATA)
    H, W = X.shape[2], X.shape[3]
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    # Seed AFTER data loading (process_data_multi_res resets the global torch seed); this controls
    # model init + batch order so seeds genuinely vary.
    torch.manual_seed(seed); np.random.seed(seed)
    # TEMPORAL holdout: train on the earlier 80% of months, validate on the later 20%. A random
    # split would leak across the strong temporal autocorrelation of SST and inflate AUROC.
    ntr = int(0.8 * len(ds))
    tr = torch.utils.data.Subset(ds, range(ntr))
    va = torch.utils.data.Subset(ds, range(ntr, len(ds)))
    tl = DataLoader(tr, batch_size=128, shuffle=True)
    vl = DataLoader(va, batch_size=256)

    gate = None if tile is None else MaskGate(H, W, tile).to(DEVICE)
    model = DeepCNN(gate).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    best = {"f1": -1}
    best_map = None
    for ep in range(epochs):
        if gate is not None:
            gate.tau = 2.0 - 1.99 * (ep / max(1, epochs - 1))
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.view(-1, 1).to(DEVICE)
            xb = xb * (torch.rand_like(xb) > 0.20) / 0.80          # multiplicative input dropout
            loss = bce(model(xb), yb)
            if gate is not None:
                loss = loss + gate.penalty(lam_sp, lam_tv)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for xb, yb in vl:
                p = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().ravel()
                ps.append(p); ys.append(yb.numpy().ravel())
        ps, ys = np.concatenate(ps), np.concatenate(ys)
        f1 = f1_score(ys, ps > 0.5, zero_division=0)
        if f1 >= best["f1"]:
            auroc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
            best = {"acc": accuracy_score(ys, ps > 0.5), "f1": f1, "auroc": auroc}
            best_map = gate.importance() if gate is not None else None
    return best, best_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resolution", type=int, default=4)
    ap.add_argument("--lam-tv", type=float, default=0.2, help="TV weight (0.2 selected by the sweep)")
    args = ap.parse_args()

    GATES = [("none", None), ("pixel", 1), ("tile_2x2", 2), ("tile_4x4", 4)]
    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    rows, mask_maps = [], {}
    t0 = time.time()

    for lead in args.leads:
        corr = np.abs(ref[f"lead{lead}_corr"])
        for name, tile in GATES:
            accs, f1s, aurocs, maps = [], [], [], []
            for s in range(args.seeds):
                st = time.time()
                m, mp = train_eval(lead, tile, s, args.epochs, args.resolution, lam_tv=args.lam_tv)
                accs.append(m["acc"]); f1s.append(m["f1"]); aurocs.append(m["auroc"])
                if mp is not None:
                    maps.append(mp)
                dt = time.time() - st
            agree = float("nan")
            if maps:
                avg = np.mean(maps, 0)
                mask_maps[f"lead{lead}_{name}"] = avg
                agree = float(np.corrcoef(avg.ravel(), corr.ravel())[0, 1])
            rows.append({"lead": lead, "gate": name,
                         "acc": np.mean(accs), "acc_sd": np.std(accs),
                         "f1": np.mean(f1s), "auroc": np.nanmean(aurocs),
                         "agree_corr": agree, "sec_per_run": dt})
            print(f"  lead {lead} {name:<9} acc={np.mean(accs):.3f} f1={np.mean(f1s):.3f} "
                  f"auroc={np.nanmean(aurocs):.3f} agree={agree:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "sst_masking.csv", index=False)
    np.savez(REPO_ROOT / "results" / "sst_masking_maps.npz", **mask_maps)

    # Figure: correlation reference + the three masks, per lead.
    cols = ["corr", "pixel", "tile_2x2", "tile_4x4"]
    titles = {"corr": "Correlation ref.", "pixel": "Pixel mask", "tile_2x2": "2x2 tiles", "tile_4x4": "4x4 tiles"}
    fig, axes = plt.subplots(len(args.leads), 4, figsize=(13, 3.0 * len(args.leads)), squeeze=False)
    for r, lead in enumerate(args.leads):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            if key == "corr":
                m = np.abs(ref[f"lead{lead}_corr"]); cmap = "viridis"
            else:
                m = mask_maps.get(f"lead{lead}_{key}"); cmap = "viridis"
            if m is not None:
                ax.imshow(m, cmap=cmap, extent=EXTENT, aspect="auto", origin="upper")
            ax.set_title(f"{titles[key]} — lead {lead}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Learned input masks vs. statistical correlation reference (El Niño, 30x90 field)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(WRITEUP / "sst_masks_vs_reference.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"\nDevice: {DEVICE}.  Total {time.time()-t0:.0f}s")
    print(df.to_string(index=False))
    print(f"\nwrote results/sst_masking.csv, results/sst_masking_maps.npz, write_up/sst_masks_vs_reference.png")


if __name__ == "__main__":
    main()
