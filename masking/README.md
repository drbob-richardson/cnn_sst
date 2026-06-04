# Masking — the paper's core method

This directory holds the code for the paper's central contribution: a **learnable
input mask / gating mechanism** (`PixelMaskGate`) trained jointly with a CNN and
regularized for sparsity + total variation. This is distinct from the rest of the
repo, which implements the **earlier** El Niño + multi-resolution + *post-hoc*
saliency experiments.

## What's here

| File | Paper section | Status |
|------|---------------|--------|
| `simulation_study_1.py` | §3 — Simulation Study: Structured Spatial Signals (Table 1, Fig 1) | ✅ Complete. 24×48 synthetic images, circular signal/distractor regions, XOR labels, 100 seeds. `PixelMaskGate` + `SmallCNN` vs. baseline; `captum` Saliency for comparison; IoU / saliency-mass metrics. (Was `~/Downloads/sim_loop.py`.) |
| `collect_elnino_results.py` | §6 — El Niño (aggregates the per-lead/per-gate metrics) | ✅ Aggregator only. Scans `results/lead_{lead}_{gate}/metrics_*.txt` and writes tidy CSV/JSON summaries + leaderboard. (Was `~/Downloads/collect_results.py`.) |

## ⚠️ Known-missing code (referenced by the paper but not found on disk)

The following were **not recoverable** when this repo was organized — they need to
be located or rebuilt before §4 and §6 can be reproduced:

1. **`learn_pixel_masks_loop_serial.py`** — the El Niño masking driver that trains
   the `DeepCNN` (5 conv: 1→32→64→64→128→128) with pixel / 2×2 / 4×4 gates over
   leads {3,6,9,12,15} and writes the `results/lead_k_<gate>/` folders that
   `collect_elnino_results.py` reads. Named in that script's docstring.
2. **Simulation Study 2 code** (§4) — 48×96 crosses, Small/Medium/Dilated
   backbones × {None, Location, Tiled 2×2, Tiled 4×4} gates (Tables 2–3, Figs 2–4).
3. **Figure-generating scripts** for the paper's mask-heatmap / Grad-CAM panels
   (Figs 3, 5, 6, 7).

If these live in a Colab, on the compute server, or another machine, drop them in
here and update this table.
