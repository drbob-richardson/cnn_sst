"""Combined sparsity-/TV-lambda sweep on Simulation-Study-1.

Resolves two questions left open by the ±TV factorial:
  (a) MODULAR: does the TV add-on help L0 / STG once we move OFF the IoU ceiling
      (i.e. at higher sparsity-lambda, where their localization is weaker)?
  (b) FAIRNESS: compare attribution faithfulness at MATCHED predictive performance
      by tracing each method's (F1, IoU_sig) frontier across lambda.

For L0 and STG we sweep the sparsity weight x {TV off, TV on}. For our sigmoid mask we
sweep the sparsity weight lambda_sp x {TV off, TV on}. Reuses gates/trainers from
`sim_attribution_benchmark` and data/metrics from `simulation_study_1`.

Usage (from repo root):
    .venv/bin/python masking/sim_lambda_sweep.py --seeds 10 --epochs 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from masking.sim_attribution_benchmark import L0Gate, STGGate, train_gated, train_ours, DEVICE  # noqa: E402
from masking.simulation_study_1 import (  # noqa: E402
    simulate_spatial_input, generate_labels, signal_regions,
    region_A, region_B, region_distractor_1, region_distractor_2,
    SmallCNN, PixelMaskGate, compute_interpretability_metrics, eval_metrics, H, W, N,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TV_ON = 5e-2
REGIONS = (region_A, region_B, region_distractor_1, region_distractor_2)


def ours_imap(gate):
    return torch.sigmoid(gate.z_main / gate.tau).detach().view(H, W).cpu().numpy()


def run_seed(seed, epochs, lam_grid, sp_grid):
    np.random.seed(seed); torch.manual_seed(seed)
    X = simulate_spatial_input(N, H, W, signal_regions)
    y, _ = generate_labels(X, region_A, region_B)
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    ntr = int(0.8 * N)
    tr, te = random_split(ds, [ntr, N - ntr])
    tr_loader = DataLoader(tr, batch_size=64, shuffle=True)
    te_loader = DataLoader(te, batch_size=128)
    rows = []

    def record(method, lam, tv, imap, model):
        m = compute_interpretability_metrics(imap, *REGIONS)
        acc, f1 = eval_metrics(model, te_loader)
        rows.append({"seed": seed, "method": method, "lam": lam, "tv": tv,
                     "iou_true": m["iou_true"], "iou_distractor": m["iou_distractor"],
                     "acc": acc, "f1": f1})

    for tv in (0.0, TV_ON):
        for lam in lam_grid:
            g = L0Gate(H, W).to(DEVICE)
            mdl = train_gated(SmallCNN(g).to(DEVICE), tr_loader, epochs, g, lam, tv_lambda=tv)
            record("l0", lam, tv, g.importance_map(), mdl)

            g = STGGate(H, W).to(DEVICE)
            mdl = train_gated(SmallCNN(g).to(DEVICE), tr_loader, epochs, g, lam, tv_lambda=tv)
            record("stg", lam, tv, g.importance_map(), mdl)

        for lam in sp_grid:
            g = PixelMaskGate(H, W).to(DEVICE)
            mdl = train_ours(SmallCNN(g).to(DEVICE), g, tr_loader, epochs, lambda_sp=lam, lambda_tv=tv)
            record("ours", lam, tv, ours_imap(g), mdl)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=str, default="results/sim_lambda_sweep.csv")
    args = ap.parse_args()

    lam_grid = [0.25, 0.5, 1.0, 2.0, 4.0]      # sparsity weight for L0/STG reg_loss
    sp_grid = [0.02, 0.05, 0.1, 0.3]           # lambda_sp for our sigmoid mask

    rows = []
    for s in range(args.seeds):
        rows += run_seed(s, args.epochs, lam_grid, sp_grid)
        print(f"  seed {s} done")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    def agg(sub):
        return sub.groupby(["lam"]).agg(iou=("iou_true", "mean"), iou_d=("iou_distractor", "mean"),
                                        f1=("f1", "mean")).reset_index()

    # ---- (a) MODULAR: per-lambda ±TV effect for L0 / STG (controls for the ceiling) ----
    print("\n=== (a) MODULAR — does TV help L0/STG at each sparsity level? ===")
    print(f"{'method':<6}{'lam':>6}{'IoU_noTV':>10}{'IoU_TV':>9}{'delta':>8}{'p':>9}")
    for method in ("l0", "stg"):
        for lam in lam_grid:
            a = df[(df.method == method) & (df.lam == lam) & (df.tv == 0.0)].sort_values("seed")["iou_true"].values
            b = df[(df.method == method) & (df.lam == lam) & (df.tv == TV_ON)].sort_values("seed")["iou_true"].values
            try:
                p = wilcoxon(a, b).pvalue if np.any(a - b != 0) else float("nan")
            except Exception:
                p = float("nan")
            print(f"{method:<6}{lam:>6}{a.mean():>10.3f}{b.mean():>9.3f}{b.mean()-a.mean():>+8.3f}{p:>9.4f}")

    # ---- (b) FAIRNESS: faithfulness at matched predictive performance ----
    print("\n=== (b) FRONTIER — (F1, IoU_sig) across lambda, TV on ===")
    print(f"{'method':<6}{'lam':>6}{'F1':>7}{'IoU_sig':>9}{'IoU_dis':>9}")
    for method in ("l0", "stg", "ours"):
        sub = df[(df.method == method) & (df.tv == TV_ON)]
        for _, r in agg(sub).iterrows():
            print(f"{method:<6}{r['lam']:>6}{r['f1']:>7.3f}{r['iou']:>9.3f}{r['iou_d']:>9.3f}")

    print("\n=== (b) MATCHED-F1 — mean IoU_sig among configs with F1 in [0.45, 0.62] (TV on) ===")
    band = df[(df.tv == TV_ON) & (df.f1 >= 0.45) & (df.f1 <= 0.62)]
    if len(band):
        for method in ("l0", "stg", "ours"):
            sb = band[band.method == method]
            if len(sb):
                print(f"  {method:<6} IoU_sig={sb['iou_true'].mean():.3f}  IoU_dis={sb['iou_distractor'].mean():.3f}  "
                      f"(n={len(sb)}, mean F1={sb['f1'].mean():.3f})")
    else:
        print("  (no configs landed in the F1 band — widen it after inspecting the frontier)")

    print(f"\nPer-(seed,config) rows written to {out}")


if __name__ == "__main__":
    main()
