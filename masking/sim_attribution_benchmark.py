"""Attribution-faithfulness benchmark on the synthetic Simulation-Study-1 data.

Compares the proposed TV+sparsity input mask against:
  - post-hoc attributions (Saliency, Integrated Gradients, Grad-CAM, Gradient-SHAP)
    applied to an ungated SmallCNN, and
  - learned in-model gates (L0 / hard-concrete, STG, spatial attention),
  - the ablation: our mask WITH vs WITHOUT the total-variation term.

Every method yields a single H x W importance map, scored against the KNOWN signal
and distractor regions (IoU + saliency-mass). We also record predictive accuracy/F1
and training time per method, and run a paired Wilcoxon test (ours+TV vs each method)
across seeds.

Reuses data generation, the SmallCNN backbone, the PixelMaskGate, and the metric from
`masking/simulation_study_1.py`. Runs on CPU (data is tiny; avoids MPS/captum op gaps).

Usage (from repo root):
    .venv/bin/python masking/sim_attribution_benchmark.py --seeds 15 --epochs 50
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset, random_split
from captum.attr import Saliency, IntegratedGradients, GradientShap, LayerGradCam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.simulation_study_1 import (  # noqa: E402
    simulate_spatial_input, generate_labels, signal_regions,
    region_A, region_B, region_distractor_1, region_distractor_2,
    SmallCNN, compute_interpretability_metrics, eval_metrics, H, W, N,
)

DEVICE = torch.device("cpu")          # tiny data; CPU is fastest & most reliable here
REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- gates
class L0Gate(nn.Module):
    """Per-location hard-concrete (L0) gate (Louizos et al. 2018). Sparsity only."""
    beta, gamma, zeta = 2.0 / 3.0, -0.1, 1.1

    def __init__(self, H, W):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(H * W))
        self.H, self.W = H, W

    def _z(self, sample):
        if sample:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0, 1)

    def forward(self, x):
        z = self._z(self.training)
        return x * z.view(1, 1, self.H, self.W).to(x.device), z

    def reg_loss(self):
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta)).mean()

    def spatial_map_for_tv(self, x):
        return self._z(False).view(self.H, self.W)          # expected gate map

    def importance_map(self, X=None):
        return self._z(False).detach().view(self.H, self.W).cpu().numpy()


class STGGate(nn.Module):
    """Per-location stochastic gate (Yamada et al. 2020). Sparsity only, no spatial coupling."""
    sigma = 0.5

    def __init__(self, H, W):
        super().__init__()
        self.mu = nn.Parameter(torch.full((H * W,), 0.5))
        self.H, self.W = H, W

    def forward(self, x):
        z = (self.mu + self.sigma * torch.randn_like(self.mu)).clamp(0, 1) if self.training else self.mu.clamp(0, 1)
        return x * z.view(1, 1, self.H, self.W).to(x.device), z

    def reg_loss(self):
        # expected number of open gates: Phi(mu/sigma)
        return (0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))).mean()

    def spatial_map_for_tv(self, x):
        return self.mu.clamp(0, 1).view(self.H, self.W)

    def importance_map(self, X=None):
        return self.mu.clamp(0, 1).detach().view(self.H, self.W).cpu().numpy()


class SpatialAttnGate(nn.Module):
    """CBAM-style input-level spatial attention: conv -> sigmoid map applied to input."""
    def __init__(self, H, W):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid(),
        )
        self.H, self.W = H, W

    def forward(self, x):
        m = self.net(x)
        return x * m, m

    def reg_loss(self):
        return torch.tensor(0.0)

    def spatial_map_for_tv(self, x):
        return self.net(x).mean(0).squeeze()                # batch-mean attention map

    def importance_map(self, X):
        self.eval()
        with torch.no_grad():
            m = self.net(X.to(DEVICE))
        return m.mean(0).squeeze().cpu().numpy()


# --------------------------------------------------------------------------- training
def _tv(m2d):
    return (m2d[:, 1:] - m2d[:, :-1]).abs().sum() + (m2d[1:, :] - m2d[:-1, :]).abs().sum()


def train_gated(model, loader, epochs, gate=None, lambda_reg=0.0, tv_lambda=0.0):
    """Generic trainer: BCE + lambda_reg * gate.reg_loss() + tv_lambda * TV(gate map).
    Used for baseline/L0/STG/attention; tv_lambda>0 adds the modular TV smoothness prior."""
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.float().to(DEVICE), yb.view(-1, 1).float().to(DEVICE)
            loss = loss_fn(model(xb), yb)
            if gate is not None and lambda_reg > 0:
                loss = loss + lambda_reg * gate.reg_loss()
            if gate is not None and tv_lambda > 0:
                loss = loss + tv_lambda * _tv(gate.spatial_map_for_tv(xb)) / (H * W)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_ours(model, gate, loader, epochs, lambda_sp=1e-1, lambda_tv=5e-2):
    """Our PixelMaskGate trainer: sparsity + (vectorized) TV penalty, with tau annealing.
    Vectorized TV is numerically identical to the looped version in simulation_study_1."""
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        gate.tau = 2 - 1.9 * (epoch / epochs)
        for xb, yb in loader:
            xb, yb = xb.float().to(DEVICE), yb.view(-1, 1).float().to(DEVICE)
            loss = loss_fn(model(xb), yb)
            m = torch.sigmoid(gate.z_main / gate.tau)
            loss = loss + lambda_sp * (m ** 2).mean() + lambda_tv * _tv(m.view(H, W)) / (H * W)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# --------------------------------------------------------------------------- post-hoc maps
def posthoc_map(method, X, kind):
    """Mean |attribution| over X -> (H, W) importance map."""
    inp = X.clone().detach().to(DEVICE).requires_grad_(True)
    base = torch.zeros_like(inp)
    if kind == "gshap":
        attr = method.attribute(inp, baselines=base, target=0, n_samples=8, stdevs=0.1)
    elif kind == "ig":
        attr = method.attribute(inp, baselines=base, target=0)
    elif kind == "gradcam":
        attr = F.relu(method.attribute(inp, target=0))
    else:  # saliency
        attr = method.attribute(inp, target=0)
    a = attr.detach().abs()
    return a.mean(dim=tuple(range(a.dim() - 2))).cpu().numpy()   # collapse batch+channel -> (H,W)


# --------------------------------------------------------------------------- one seed
def run_seed(seed, epochs, n_attr):
    np.random.seed(seed); torch.manual_seed(seed)
    X = simulate_spatial_input(N, H, W, signal_regions)
    y, _ = generate_labels(X, region_A, region_B)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    ntr = int(0.8 * N)
    tr, te = random_split(ds, [ntr, N - ntr])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    te_loader = DataLoader(te, batch_size=128)
    Xte = torch.stack([te[i][0] for i in range(len(te))]).float()[:n_attr]

    regions = (region_A, region_B, region_distractor_1, region_distractor_2)
    rows = []

    def record(method, imap, model, t):
        m = compute_interpretability_metrics(imap, *regions)
        acc, f1 = eval_metrics(model, te_loader)
        rows.append({"seed": seed, "method": method, **m, "acc": acc, "f1": f1, "train_s": t})

    # ---- ungated baseline (shared by all post-hoc methods) ----
    t0 = time.time(); base = train_gated(SmallCNN().to(DEVICE), tr_loader, epochs); tb = time.time() - t0
    for name, kind, ctor in [
        ("saliency", "saliency", lambda: Saliency(base)),
        ("integrated_grad", "ig", lambda: IntegratedGradients(base)),
        ("grad_cam", "gradcam", lambda: LayerGradCam(base, base.conv2)),
        ("gradient_shap", "gshap", lambda: GradientShap(base)),
    ]:
        record(name, posthoc_map(ctor(), Xte, kind), base, tb)

    # ---- learned in-model gates, each WITHOUT and WITH the modular TV add-on ----
    for name, ctor, lam in [("l0_gate", lambda: L0Gate(H, W), 1.0),
                            ("stg_gate", lambda: STGGate(H, W), 1.0),
                            ("spatial_attn", lambda: SpatialAttnGate(H, W), 0.0)]:
        for tag, tv in [("", 0.0), ("_TV", 5e-2)]:
            gate = ctor().to(DEVICE)
            t0 = time.time()
            model = train_gated(SmallCNN(gate).to(DEVICE), tr_loader, epochs, gate, lam, tv_lambda=tv)
            record(name + tag, gate.importance_map(Xte), model, time.time() - t0)

    # ---- ours: pixel mask with and without TV ----
    from masking.simulation_study_1 import PixelMaskGate
    for name, ltv in [("ours_TV", 5e-2), ("ours_noTV", 0.0)]:
        gate = PixelMaskGate(H, W).to(DEVICE)
        t0 = time.time(); model = train_ours(SmallCNN(gate).to(DEVICE), gate, tr_loader, epochs, lambda_tv=ltv)
        imap = torch.sigmoid(gate.z_main / gate.tau).detach().view(H, W).cpu().numpy()
        record(name, imap, model, time.time() - t0)

    return rows


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-attr", type=int, default=100, help="# test images for post-hoc maps")
    ap.add_argument("--out", type=str, default="results/sim_attribution_benchmark.csv")
    args = ap.parse_args()

    all_rows = []
    for s in range(args.seeds):
        all_rows += run_seed(s, args.epochs, args.n_attr)
        print(f"  seed {s} done")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    order = ["saliency", "integrated_grad", "grad_cam", "gradient_shap",
             "l0_gate", "l0_gate_TV", "stg_gate", "stg_gate_TV",
             "spatial_attn", "spatial_attn_TV", "ours_noTV", "ours_TV"]
    print(f"\n=== Attribution faithfulness: mean over {args.seeds} seeds ===")
    print(f"{'method':<16}{'IoU_sig':>8}{'IoU_dis':>8}{'mass_sig':>9}{'mass_dis':>9}{'acc':>7}{'f1':>7}{'train_s':>9}{'p_vs_ours':>10}")
    ref = df[df.method == "ours_TV"].sort_values("seed")["iou_true"].values
    for mth in order:
        sub = df[df.method == mth].sort_values("seed")
        iou = sub["iou_true"].values
        if mth != "ours_TV" and len(iou) == len(ref) and np.any(iou - ref != 0):
            try:
                p = wilcoxon(ref, iou).pvalue
            except Exception:
                p = float("nan")
            pstr = f"{p:>10.4f}"
        else:
            pstr = f"{'—':>10}"
        print(f"{mth:<16}{sub['iou_true'].mean():>8.3f}{sub['iou_distractor'].mean():>8.3f}"
              f"{sub['mass_true'].mean():>9.3f}{sub['mass_distractor'].mean():>9.3f}"
              f"{sub['acc'].mean():>7.3f}{sub['f1'].mean():>7.3f}{sub['train_s'].mean():>9.1f}{pstr}")
    # ---- headline: per-method TV add-on effect (within-method ±TV ablation, fair by construction) ----
    print("\n=== TV add-on effect per method (IoU_sig: no-TV -> +TV) ===")
    print(f"{'method':<14}{'noTV':>8}{'+TV':>8}{'delta':>8}{'p(Wilcoxon)':>13}")
    for a, b in [("l0_gate", "l0_gate_TV"), ("stg_gate", "stg_gate_TV"),
                 ("spatial_attn", "spatial_attn_TV"), ("ours_noTV", "ours_TV")]:
        ia = df[df.method == a].sort_values("seed")["iou_true"].values
        ib = df[df.method == b].sort_values("seed")["iou_true"].values
        try:
            p = wilcoxon(ia, ib).pvalue if np.any(ia - ib != 0) else float("nan")
        except Exception:
            p = float("nan")
        label = a.replace("_noTV", "").replace("ours", "ours_mask")
        print(f"{label:<14}{ia.mean():>8.3f}{ib.mean():>8.3f}{ib.mean() - ia.mean():>+8.3f}{p:>13.4f}")

    print(f"\nPer-seed rows written to {out}")
    print("Higher IoU_sig / mass_sig = better (finds true signal); lower IoU_dis / mass_dis = better (ignores distractors).")
    print("p_vs_ours = paired Wilcoxon on IoU_sig vs ours_TV.")


if __name__ == "__main__":
    main()
