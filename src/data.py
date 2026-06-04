"""Shared SST data pipeline for the El Niño forecasting experiments.

This module centralizes the data loading / preprocessing that was previously
duplicated inside ``elnino_prediction_simple.py``. Every experiment script in
``experiments/`` and ``saliency/`` imports :func:`process_data_multi_res` from
here.

Data file
---------
The pipeline expects a NOAA SST NetCDF file. By default it looks for
``<repo>/data/sst.mon.mean.nc`` (the full NOAA OISST monthly-mean field). The
89 MB ``data/sst_data.nc`` shipped with the repo is the same field clipped to
the analysis region and can be passed explicitly via ``file_path``. See
``data/README.md`` for download instructions.
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# Repository root is one level up from this file (src/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SST_FILE = REPO_ROOT / "data" / "sst.mon.mean.nc"

# Niño 3.4 region (used to build the binary El Niño label).
NINO34_LAT = slice(5, -5)
NINO34_LON = slice(190, 240)
# Broader equatorial Pacific region fed to the CNN (≈30 x 90 grid).
INPUT_LAT = slice(15, -15)
INPUT_LON = slice(170, 260)

# El Niño label threshold: Niño 3.4 anomaly (°C) above which we flag an event.
EVENT_THRESHOLD = 0.5


class SSTDataset(Dataset):
    """Minimal ``(data, label)`` dataset wrapping numpy arrays as float tensors."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def process_data_multi_res(lead_time, resolution=1, seed=1975, file_path=None):
    """Build (input, label) arrays for El Niño classification at a given lead/resolution.

    Parameters
    ----------
    lead_time : int
        Forecast horizon in months. The label is whether the Niño 3.4 anomaly
        ``lead_time`` months ahead exceeds :data:`EVENT_THRESHOLD`.
    resolution : int, default 1
        Spatial down-sampling stride applied to the input field (``isel`` step).
        ``1`` keeps full resolution; larger values coarsen the grid.
    seed : int, default 1975
        Seed for ``torch`` (kept for reproducibility with the original code).
    file_path : str or Path, optional
        Override for the SST NetCDF file. Defaults to :data:`DEFAULT_SST_FILE`.

    Returns
    -------
    data : np.ndarray
        Float array of shape ``(T - lead_time, 1, H, W)``.
    labels : np.ndarray
        Binary array of shape ``(T - lead_time,)``.
    """
    torch.manual_seed(seed)

    file_path = Path(file_path) if file_path is not None else DEFAULT_SST_FILE
    # Normalize axis order so the slices below work regardless of how the file
    # stores its coordinates: latitude descending (15° -> -15°), longitude
    # ascending. The shipped data/sst_data.nc stores latitude ascending, which
    # would make slice(15, -15) return an empty range without this.
    sst = xr.open_dataset(file_path)["sst"].sortby("lat", ascending=False).sortby("lon")
    sst_broad = sst.sel(lat=INPUT_LAT, lon=INPUT_LON)

    sst_nino34 = sst_broad.sel(lat=NINO34_LAT, lon=NINO34_LON)

    # Monthly anomalies (deviation from the monthly climatology).
    sst_anomalies = sst_broad.groupby("time.month") - sst_broad.groupby("time.month").mean(dim="time")
    sst_nino34_anomalies = (
        sst_nino34.groupby("time.month") - sst_nino34.groupby("time.month").mean(dim="time")
    )

    # Binary label per time step: El Niño event ``lead_time`` months ahead.
    T = len(sst_nino34_anomalies["time"])
    labels = [
        1 if sst_nino34_anomalies.isel(time=t + lead_time).mean(dim=["lat", "lon"]).values > EVENT_THRESHOLD else 0
        for t in range(T - lead_time)
    ]

    # Down-sample the input field spatially according to ``resolution``.
    sst_downsampled = sst_anomalies.isel(lat=slice(0, None, resolution), lon=slice(0, None, resolution))
    data = np.array([sst_downsampled.isel(time=t).values for t in range(T - lead_time)])

    # Add a channel dimension for CNN compatibility: (T, 1, H, W).
    data = data[:, np.newaxis, :, :]

    return data, np.array(labels)
