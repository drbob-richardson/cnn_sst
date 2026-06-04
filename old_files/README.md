# Archived / superseded files

These files are kept for provenance but are **not part of the active pipeline**.
Nothing in the repo imports them.

| File | What it is | Why archived |
|------|-----------|--------------|
| `cnn_functions.py` | A helper module defining a window-based `process_data`, `SimpleCNN`, `SimpleCNN3D`, and `SSTDataset`. Note its `process_data` references a global `sst_broad` that the module never defines. | **Dead code** — no script imports it. The same model classes are defined inline in `experiments/window_lead_3dcnn.py`, which is the version actually used. |
| `resloop_improved_variant.py` | An improved version of the single-resolution sweep (was `Dynamic_Downscaling/resloop.py`). Adds per-epoch best-model checkpointing on out-of-sample F1 and trains 100 epochs. | **Superseded duplicate** with a real improvement. The active `experiments/single_resolution_sweep.py` is the simpler original that produced `results/single_resolution_sweep_results.csv`. **Worth merging** the best-model-tracking loop into the active script later. |
| `Masked_Input_CNNs-7.textClipping` | A macOS text clipping containing only the paper's title (no content). | Not code or data; kept only because it was in the repo root. The actual paper PDF lives in `~/Downloads/Masked_Input_CNNs-7.pdf`. |

## Also removed during the reorganization (recoverable from git history)

- **`Dynamic_Downscaling/` and `Saliency_Project/`** — these were parallel working
  copies. Every file in them duplicated a root-level file (verified by checksum)
  except the improved `resloop.py` (saved here) and three notebooks
  (`results_plotter.ipynb`, `saliency_movie_2.ipynb`) plus `make_saliency_movie.py`,
  which were promoted to `notebooks/` and `saliency/`.
- **A nested `cnn_sst/` clone** — a byte-identical clone of this same GitHub repo
  (`drbob-richardson/cnn_sst`, same commit history) that had been dropped into the
  working directory. Removed because it added nothing; re-clone from GitHub if needed.
- **`__pycache__/`, `.DS_Store`, empty `Data/` and `from_server/` folders.**
