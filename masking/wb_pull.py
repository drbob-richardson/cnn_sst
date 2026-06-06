"""One-time cached download of a manageable WeatherBench-2 slice.

Throughput from this environment is slow (~0.5 MB/s), so we pull a small field once and cache it
locally; all experiments then run on the cache. Geopotential @ 500 hPa (Z500), the canonical
WeatherBench field, on the 64x32 grid, 6-hourly, ~12 years -> ~16k samples (vs ENSO's ~510).
"""

import time
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "wb_z500.npz"
PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

ds = xr.open_zarr(PATH, storage_options={"token": "anon"}, chunks={"time": 4096})
z = (ds["geopotential"].sel(level=500)
     .sel(time=slice("2010-01-01", "2021-12-31"))
     .transpose("time", "latitude", "longitude"))
print(f"pulling Z500 {z.shape} ...")
t = time.time()
arr = z.values.astype("float32")
print(f"  pulled in {time.time()-t:.0f}s  ({arr.nbytes/1e6:.0f} MB)")
np.savez(OUT, z=arr,
         time=z["time"].values.astype("datetime64[h]").astype("int64"),
         month=z["time"].dt.month.values.astype("int16"),
         lat=ds["latitude"].values.astype("float32"),
         lon=ds["longitude"].values.astype("float32"))
print(f"saved -> {OUT}  shape {arr.shape}")
