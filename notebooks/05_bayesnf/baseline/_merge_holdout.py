"""Merge holdout FFRK features with rainfall + static metadata.

Produces ``bayesnf_holdout_test.parquet`` with the SAME 39-column schema as
``bayesnf_fold0_test.parquet`` and uploads it to S3.

Inputs:
  * ``outputs/holdout_station_ids.json`` — 492 holdout station IDs
  * ``data/rekis/`` — ReKIS daily precipitation + Stationsliste
  * ``data/dem/`` — Copernicus DEM tiles (for elevation_m)
  * ``data/soilgrids/`` — SoilGrids rasters (6 variables × 3 depths)
  * ``s3://thesis-data-ismaktam/bayesnf/test/features.parquet`` — FFRK
    features (idw, gos, svd_00..svd_20) for holdout stations

Outputs:
  * ``results/bayesnf/data/bayesnf_holdout_test.parquet`` (local)
  * ``s3://thesis-data-ismaktam/bayesnf/data/bayesnf_holdout_test.parquet``

Run:
    python notebooks/05_bayesnf/baseline/_merge_holdout.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

# project imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thesis.config import Config
from thesis.data.dem import DEMSource
from thesis.data.rekis import ReKISSource
from thesis.data.soilgrids import SoilGridsSource

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE_START = "1961-01-01"
DATE_END   = "2023-12-31"

HOLDOUT_IDS_PATH = PROJECT_ROOT / "outputs" / "holdout_station_ids.json"
FFRK_S3_KEY      = "bayesnf/test/features.parquet"
FFRK_LOCAL_CACHE = Path("/tmp") / "bayesnf_test_features.parquet"

OUT_LOCAL = PROJECT_ROOT / "results" / "bayesnf" / "data" / "bayesnf_holdout_test.parquet"
OUT_S3_BUCKET = "thesis-data-ismaktam"
OUT_S3_KEY    = "bayesnf/data/bayesnf_holdout_test.parquet"

SOIL_VARS = ("bulk_density", "clay", "sand", "silt", "soc", "water_10kpa")

# Column order must match bayesnf_fold0_test.parquet exactly.
FOLD_COL_ORDER = [
    "datetime", "latitude", "longitude", "x_proj", "y_proj", "elevation_m",
    "idw", "gos",
    *(f"svd_{i:02d}" for i in range(21)),
    "bulk_density", "clay", "sand", "silt", "soc", "water_10kpa",
    "rainfall", "rainfall_int", "station_id", "fold",
]
assert len(FOLD_COL_ORDER) == 39, f"expected 39 cols, got {len(FOLD_COL_ORDER)}"


def main() -> None:
    t0 = time.time()
    cfg = Config()

    # 1. Holdout station IDs --------------------------------------------------
    holdout_ids: list[str] = json.loads(HOLDOUT_IDS_PATH.read_text())
    print(f"[1/8] holdout stations: {len(holdout_ids)}")

    # 2. Rainfall + coords (lat/lon/elevation_m from Stationsliste) ----------
    print(f"[2/8] loading ReKIS rainfall for {DATE_START}..{DATE_END}")
    rekis = ReKISSource(cfg)
    rain = rekis.load(DATE_START, DATE_END, exclude_holdout=False)
    rain = rain[rain["station_id"].isin(holdout_ids)].reset_index(drop=True)
    rain = rain.dropna(subset=["precip_mm"]).reset_index(drop=True)
    print(f"      rows after NaN drop: {len(rain):,}  unique stations: "
          f"{rain['station_id'].nunique()}")

    # 3. Static per-station table --------------------------------------------
    print("[3/8] building per-station static frame")
    stn = (
        rain[["station_id", "lon", "lat"]]
        .drop_duplicates("station_id")
        .reset_index(drop=True)
    )
    # x_proj / y_proj (EPSG:4326 → EPSG:3035)
    tx = Transformer.from_crs("EPSG:4326", cfg.study_area.target_crs, always_xy=True)
    stn["x_proj"], stn["y_proj"] = tx.transform(stn["lon"].values, stn["lat"].values)

    # 4. DEM elevation (overrides Stationsliste 'Hoehe' to match fold parquets)
    print("[4/8] sampling DEM elevation")
    dem = DEMSource(cfg)
    stn["elevation_m"] = dem.sample_at_points(
        stn["lon"].values, stn["lat"].values,
    ).astype(np.float64)

    # 5. SoilGrids: depth-averaged per variable ------------------------------
    print("[5/8] sampling SoilGrids (6 vars, depth-averaged)")
    for var in SOIL_VARS:
        src = SoilGridsSource(cfg, variable=var)  # depth=None → depth average
        stn[var] = src.sample_at_points(stn["lon"].values, stn["lat"].values)
        print(f"      {var:14s}  mean={stn[var].mean():.2f}  "
              f"nan={stn[var].isna().sum()}")

    # 6. FFRK features --------------------------------------------------------
    print("[6/8] loading FFRK features")
    if not FFRK_LOCAL_CACHE.exists():
        print(f"      downloading s3://{OUT_S3_BUCKET}/{FFRK_S3_KEY} -> "
              f"{FFRK_LOCAL_CACHE}")
        boto3.client("s3", region_name="eu-north-1").download_file(
            OUT_S3_BUCKET, FFRK_S3_KEY, str(FFRK_LOCAL_CACHE),
        )
    ffrk = pd.read_parquet(FFRK_LOCAL_CACHE)
    ffrk = ffrk.rename(columns={"date": "datetime"})
    ffrk["datetime"] = pd.to_datetime(ffrk["datetime"])
    ffrk = ffrk[ffrk["station_id"].isin(holdout_ids)].reset_index(drop=True)
    print(f"      ffrk rows: {len(ffrk):,}  stations: "
          f"{ffrk['station_id'].nunique()}")

    # 7. Assemble the final frame --------------------------------------------
    print("[7/8] joining rainfall + static + FFRK")
    rain = rain.rename(columns={
        "date": "datetime", "lat": "latitude", "lon": "longitude",
        "precip_mm": "rainfall",
    })
    rain["datetime"] = pd.to_datetime(rain["datetime"])
    # Drop columns from rain that we will source from `stn` to avoid suffix clash.
    rain = rain.drop(columns=["latitude", "longitude", "elevation_m"], errors="ignore")

    stn_static = stn.drop(columns=["lon", "lat"]).rename(
        columns={"station_id": "station_id"},  # noop, kept for clarity
    )
    stn_static = stn_static.assign(
        latitude=stn["lat"].values, longitude=stn["lon"].values,
    )

    df = (
        ffrk.merge(rain[["station_id", "datetime", "rainfall"]],
                   on=["station_id", "datetime"], how="inner")
            .merge(stn_static, on="station_id", how="left")
    )
    df["rainfall_int"] = np.rint(10.0 * df["rainfall"]).astype(np.int32)
    df["fold"] = np.int8(-1)

    # Match fold0_test dtypes exactly
    for c in ("latitude", "longitude", "x_proj", "y_proj", "elevation_m", "rainfall"):
        df[c] = df[c].astype(np.float64)
    for c in (
        "idw", "gos",
        *(f"svd_{i:02d}" for i in range(21)),
        *SOIL_VARS,
    ):
        df[c] = df[c].astype(np.float32)
    df["fold"] = df["fold"].astype(np.int8)
    df["rainfall_int"] = df["rainfall_int"].astype(np.int32)

    # Reorder
    missing = set(FOLD_COL_ORDER) - set(df.columns)
    extra   = set(df.columns) - set(FOLD_COL_ORDER)
    if missing or extra:
        raise AssertionError(f"column mismatch: missing={missing} extra={extra}")
    df = df[FOLD_COL_ORDER]

    # Final NaN check on critical cols
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if len(nan_cols):
        print("      WARNING — NaN values per column:")
        print(nan_cols.to_string())
    print(f"      final rows: {len(df):,}  stations: {df['station_id'].nunique()}")

    # 8. Save + upload --------------------------------------------------------
    print("[8/8] writing parquet + uploading to S3")
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_LOCAL, index=False)
    sz = OUT_LOCAL.stat().st_size / 1e6
    print(f"      local file: {OUT_LOCAL}  ({sz:.1f} MB)")

    # Verify schema matches fold0_test (best-effort: just dtypes)
    schema = pq.read_schema(OUT_LOCAL)
    print(f"      schema cols: {len(schema)}")
    for fld in schema:
        print(f"        {fld.name:14s} -> {fld.type}")

    s3 = boto3.client("s3", region_name="eu-north-1")
    s3.upload_file(str(OUT_LOCAL), OUT_S3_BUCKET, OUT_S3_KEY)
    print(f"      uploaded -> s3://{OUT_S3_BUCKET}/{OUT_S3_KEY}")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
