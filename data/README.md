# Data

This project forecasts El Niño events from gridded sea-surface temperature (SST)
fields. The SST data is **not committed to git** (it is large and freely
re-downloadable) and is ignored via `.gitignore`.

## Expected files

| File | Used by | Notes |
|------|---------|-------|
| `data/sst_data.nc` | `experiments/multi_resolution_combined.py`, `experiments/window_lead_3dcnn.py` | ~89 MB. The SST field clipped to the analysis region. Present locally; reproduce as below. |
| `data/sst.mon.mean.nc` | `src/data.py` → `process_data_multi_res` (imported by most experiments) | The full NOAA monthly-mean SST field. **Not present** — download as below. |

`src/data.py` selects the analysis region itself (`lat 15°…-15°`, `lon 170°…260°`,
Niño-3.4 = `lat 5°…-5°`, `lon 190°…240°`), so it expects the **full** field
`sst.mon.mean.nc`. The clipped `sst_data.nc` was used by the two standalone
scripts above. If you only have one of the two files, point the scripts at it via
the `file_path` argument to `process_data_multi_res` (or by editing the path).

## How to obtain `sst.mon.mean.nc`

NOAA OISST / Extended Reconstructed SST monthly means, from the NOAA PSL
catalogue (e.g. NOAA OISST v2 "sst.mon.mean.nc"):

```bash
# Example (verify the current URL on the NOAA PSL data portal):
curl -L -o data/sst.mon.mean.nc \
  https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mon.mean.nc
```

The dataset must expose an `sst` variable with `time`, `lat`, `lon` dimensions
and longitudes in the 0–360° convention.
