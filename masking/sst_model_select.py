"""Defensible model selection for the El Nino SST task.

The original DeepCNN (~288k params for ~410 training months) was inherited from the paper spec and
never tuned --- heavily over-parameterized. Here we select an architecture honestly:

  3-way TEMPORAL split:  train [0,60%)  |  validation [60%,80%)  |  test [80%,100%]
  - Phase 1 (selection): train the UNGATED baseline of each candidate on train, pick the epoch by
    validation AUROC, and select the architecture with the best mean validation AUROC.
  - Phase 2 (final): with the selected architecture, train none / pixel / 2x2 / 4x4 gates (train,
    early-stopped on validation) and report metrics on the UNTOUCHED test block, plus masks and
    agreement with the correlation reference.

The test block is never used for any selection decision.

Usage:  .venv/bin/python masking/sst_model_select.py --leads 3 6
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import process_data_multi_res  # noqa: E402
from masking.sst_masking import MaskGate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITEUP = REPO_ROOT / "write_up"
DATA = REPO_ROOT / "data" / "sst_data.nc"
EXTENT = [170, 260, -15, 15]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ACT = {"relu": nn.ReLU, "gelu": nn.GELU}

# Candidate backbones: (channels, pool-after-each-conv, head hidden dims, activation).
CANDIDATES = {
    "deep_relu":   ([1, 32, 64, 64, 128, 128], False, [64, 32], "relu"),   # the incumbent (~288k)
    "medium_relu": ([1, 16, 32, 64], True, [32], "relu"),
    "small_relu":  ([1, 16, 32], True, [], "relu"),
    "tiny_relu":   ([1, 8, 16], True, [], "relu"),
    "medium_gelu": ([1, 16, 32, 64], True, [32], "gelu"),
}


class ConfigCNN(nn.Module):
    def __init__(self, channels, pool, head_dims, act, gate=None):
        super().__init__()
        self.gate = gate
        A = ACT[act]
        layers = []
        for i in range(len(channels) - 1):
            layers += [nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                       nn.BatchNorm2d(channels[i + 1]), A()]
            if pool:
                layers += [nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        dims = [channels[-1]] + head_dims
        head = []
        for i in range(len(dims) - 1):
            head += [nn.Linear(dims[i], dims[i + 1]), A(), nn.Dropout(0.3)]
        head += [nn.Linear(dims[-1], 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        if self.gate is not None:
            x, _ = self.gate(x)
        return self.head(self.features(x).mean(dim=(2, 3)))


def nparams(name):
    ch, pool, hd, act = CANDIDATES[name]
    return sum(p.numel() for p in ConfigCNN(ch, pool, hd, act).parameters())


def _auroc(model, X, y):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32).to(DEVICE))).cpu().numpy().ravel()
    return p


def fit(arch, lead, seed, tile, epochs=60, lam_sp=1e-2, lam_tv=0.2):
    """Train on [0,60%), early-stop by val AUROC on [60%,80%), evaluate on test [80%,100%]."""
    X, y = process_data_multi_res(lead, resolution=4, file_path=DATA)
    H, W = X.shape[2], X.shape[3]
    T = len(y)
    a, b = int(0.6 * T), int(0.8 * T)
    Xtr, ytr, Xva, yva, Xte, yte = X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]
    torch.manual_seed(seed); np.random.seed(seed)

    ch, pool, hd, act = CANDIDATES[arch]
    gate = None if tile is None else MaskGate(H, W, tile).to(DEVICE)
    model = ConfigCNN(ch, pool, hd, act, gate).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1).to(DEVICE)
    n = len(Xtr); bs = 128

    best = {"val_auroc": -1}
    best_map = None
    for ep in range(epochs):
        if gate is not None:
            gate.tau = 2.0 - 1.99 * (ep / max(1, epochs - 1))
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr_t[idx]
            xb = xb * (torch.rand_like(xb) > 0.20) / 0.80
            loss = bce(model(xb), ytr_t[idx])
            if gate is not None:
                loss = loss + gate.penalty(lam_sp, lam_tv)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = _auroc(model, Xva, yva)
        va = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if va > best["val_auroc"]:
            pte = _auroc(model, Xte, yte)
            best = {"val_auroc": va,
                    "test_acc": accuracy_score(yte, pte > 0.5),
                    "test_f1": f1_score(yte, pte > 0.5, zero_division=0),
                    "test_auroc": roc_auc_score(yte, pte) if len(np.unique(yte)) > 1 else float("nan")}
            best_map = gate.importance() if gate is not None else None
    return best, best_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--sel-seeds", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    import pandas as pd

    # ---------------- Phase 1: architecture selection on validation ----------------
    print("=== Phase 1: selection (ungated baseline, validation AUROC) ===")
    sel = []
    for arch in CANDIDATES:
        vs = [fit(arch, lead, s, None, args.epochs)[0]["val_auroc"]
              for lead in args.leads for s in range(args.sel_seeds)]
        sel.append({"arch": arch, "params": nparams(arch), "val_auroc": np.mean(vs), "val_sd": np.std(vs)})
        print(f"  {arch:<12} params={nparams(arch):>7}  val_AUROC={np.mean(vs):.3f} ± {np.std(vs):.3f}")
    seldf = pd.DataFrame(sel).sort_values("val_auroc", ascending=False)
    best_arch = seldf.iloc[0]["arch"]
    seldf.to_csv(REPO_ROOT / "results" / "sst_model_select_selection.csv", index=False)
    print(f"\n  SELECTED: {best_arch} ({nparams(best_arch)} params)\n")

    # ---------------- Phase 2: final results on the untouched test block ----------------
    print(f"=== Phase 2: final test-block results with {best_arch} ===")
    ref = np.load(REPO_ROOT / "results" / "sst_statistical_reference.npz")
    GATES = [("none", None), ("pixel", 1), ("tile_2x2", 2), ("tile_4x4", 4)]
    rows, maps = [], {}
    for lead in args.leads:
        corr = np.abs(ref[f"lead{lead}_corr"])
        for gname, tile in GATES:
            accs, f1s, aus, mps = [], [], [], []
            for s in range(args.final_seeds):
                m, mp = fit(best_arch, lead, s, tile, args.epochs)
                accs.append(m["test_acc"]); f1s.append(m["test_f1"]); aus.append(m["test_auroc"])
                if mp is not None:
                    mps.append(mp)
            agree = float("nan")
            if mps:
                avg = np.mean(mps, 0); maps[f"lead{lead}_{gname}"] = avg
                agree = float(np.corrcoef(avg.ravel(), corr.ravel())[0, 1])
            rows.append({"lead": lead, "gate": gname, "arch": best_arch,
                         "test_acc": np.mean(accs), "test_acc_sd": np.std(accs),
                         "test_f1": np.mean(f1s), "test_auroc": np.nanmean(aus), "agree": agree})
            print(f"  lead {lead} {gname:<9} acc={np.mean(accs):.3f} f1={np.mean(f1s):.3f} "
                  f"auroc={np.nanmean(aus):.3f} agree={agree:+.3f}")
    fdf = pd.DataFrame(rows)
    fdf.to_csv(REPO_ROOT / "results" / "sst_model_select_test.csv", index=False)
    np.savez(REPO_ROOT / "results" / "sst_masking_maps.npz", **maps)

    # regenerate the masks-vs-reference figure for the selected architecture
    cols = ["corr", "pixel", "tile_2x2", "tile_4x4"]
    titles = {"corr": "Correlation ref.", "pixel": "Pixel mask", "tile_2x2": "2x2 tiles", "tile_4x4": "4x4 tiles"}
    fig, axes = plt.subplots(len(args.leads), 4, figsize=(13, 3.0 * len(args.leads)), squeeze=False)
    for r, lead in enumerate(args.leads):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            m = np.abs(ref[f"lead{lead}_corr"]) if key == "corr" else maps.get(f"lead{lead}_{key}")
            if m is not None:
                ax.imshow(m, cmap="viridis", extent=EXTENT, aspect="auto", origin="upper")
            ax.set_title(f"{titles[key]} — lead {lead}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Learned masks vs. correlation reference (selected backbone: {best_arch})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(WRITEUP / "sst_masks_vs_reference.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"\nDevice {DEVICE}. Total {time.time()-t0:.0f}s")
    print("\nSELECTION:\n" + seldf.to_string(index=False))
    print("\nFINAL (test block):\n" + fdf.to_string(index=False))


if __name__ == "__main__":
    main()
