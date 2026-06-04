# Interpretable CNNs for Spatio-Temporal SST Forecasting

Code for **"An Interpretable Deep Learning Framework for Spatio-Temporal
Forecasting"** (R. Richardson, Dept. of Statistics, BYU). The paper proposes a
**learnable input mask / gate** that co-trains with a CNN to produce sparse,
spatially-coherent saliency maps — an *in-model* alternative to post-hoc methods
(saliency, Integrated Gradients, SHAP, Grad-CAM) — and applies it to El Niño
prediction from sea-surface temperature (SST).

> **Repository status.** This repo currently contains two bodies of work:
> 1. **The masking method** (`masking/`) — the paper's core contribution. Only
>    Simulation Study 1 and the El Niño results-aggregator are present; the El Niño
>    masking driver and Simulation Study 2 code are **missing** (see
>    [`masking/README.md`](masking/README.md)).
> 2. **The El Niño / multi-resolution + post-hoc-saliency experiments**
>    (`experiments/`, `saliency/`) — the *precursor* work. These provide the data
>    pipeline and the baseline CNNs the paper compares its masks against.
>
> The reorganization that produced this layout is documented in
> [`old_files/README.md`](old_files/README.md).

## Layout

```
.
├── src/                  Shared library code
│   ├── data.py             SST loading + Niño-3.4 labeling (process_data_multi_res)
│   └── models.py           Shared architectures (MultiResCNN)
├── masking/              ⭐ Paper's core method (learnable input mask)
│   ├── simulation_study_1.py      Paper §3 (PixelMaskGate vs baseline, 100 seeds)
│   └── collect_elnino_results.py  Paper §6 results aggregator
├── experiments/          El Niño / multi-resolution CNN experiments (precursor work)
├── saliency/             Post-hoc saliency map + movie generation
├── notebooks/            Data exploration + results/figure plotting
├── results/             *.csv metrics (one per experiment)
├── figures/             Generated saliency-map figures + movie (movie gitignored)
├── models/              Trained weights (*.pth, gitignored)
├── data/                SST NetCDF (gitignored — see data/README.md)
└── old_files/           Archived duplicates / dead / superseded code
```

## Setup

```bash
pip install -r requirements.txt
# Obtain the SST data (see data/README.md)
```

All scripts are designed to be run **from the repository root**, e.g.:

```bash
python experiments/single_resolution_sweep.py
python masking/simulation_study_1.py
```

(Each script bootstraps `sys.path` to import `src`, and reads/writes data,
results, figures, and models via repo-root-relative paths.)

## Experiments and their outputs

| Script | Model | Output | In paper? |
|--------|-------|--------|-----------|
| `masking/simulation_study_1.py` | `PixelMaskGate` + `SmallCNN` | `sim_outputs/simulation_results.csv` | **Yes — §3, Table 1, Fig 1** |
| `experiments/deeper_cnn_saliency.py` | `DeeperCNN` (4 conv + BN) + post-hoc saliency | `results/deeper_cnn_results.csv`, `figures/saliency_deeper_cnn/` | Baseline lineage of the paper's El Niño `DeepCNN` (§6) |
| `experiments/single_resolution_sweep.py` | `SimpleCNN`, lead × resolution sweep | `results/single_resolution_sweep_results.csv` | Precursor |
| `experiments/mixed_resolution.py` | `MultiResCNN` (two-branch) | `results/mixed_resolution_results.csv` | Precursor |
| `experiments/mixed_resolution_saliency.py` | `MultiResCNN` + saliency | `results/mixed_resolution_results.csv`, `figures/saliency_mixed_res/` | Precursor |
| `experiments/single_res_saliency.py` | `SingleResCNN` + saliency | `results/single_res_saliency_results.csv`, `figures/saliency_single_res/` | Precursor |
| `experiments/multi_resolution_combined.py` | `SimpleCNN`, channel-stacked resolutions | `results/multi_resolution_results.csv` | Precursor |
| `experiments/window_lead_3dcnn.py` | `SimpleCNN3D` + temporal windows | `results/window_lead_3dcnn_results.csv` | Precursor |
| `experiments/train_cnn_model.py` | `SimpleCNN`, single fit | `models/best_cnn_model.pth` | Utility |
| `experiments/train_f1_model.py` | `SimpleCNN`, best-F1 checkpoint | `models/best_model_f1.pth` | Utility |
| `saliency/multires_saliency.py` | `MultiResCNN` saliency maps | `figures/saliency_mixed_res/` | Precursor |
| `saliency/make_saliency_movie.py` | — | `figures/saliency_movie.mp4` | Utility |

## Method (paper §2, in `masking/`)

A spatial mask `m ∈ [0,1]^{H×W}` is applied to the input via a Hadamard product,
`x̃ = x ⊙ m`, then fed to a CNN. The mask comes from per-location (or per-tile)
learnable scores through a temperature-controlled sigmoid `mₖ = σ(zₖ/τ)`, with `τ`
annealed over training. The loss adds sparsity and total-variation penalties:

```
L = BCE(ŷ, y) + λ_sp · mean|mₖ|^γ + λ_TV · mean|mᵢ − mⱼ|^γ
```

## Reproducing the El Niño masking results (TODO)

Blocked on the missing `learn_pixel_masks_loop_serial.py` (see
[`masking/README.md`](masking/README.md)). Once recovered, it writes
`results/lead_{k}_{gate}/` folders which `masking/collect_elnino_results.py`
aggregates into summary CSVs.
