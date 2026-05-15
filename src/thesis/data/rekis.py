"""ReKIS / DWD weather station loader (RR.csv + Stationsliste.txt)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thesis.cache import make_memory
from thesis.config import Config


class ReKISSource:
    """Loads daily precipitation from local rekis CSV files."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._rekis_dir = Path(cfg.paths.root) / "rekis"
        memory = make_memory(cfg)
        # The parse step is expensive (wide→long on 60 years × 250 stations).
        # Cache it so it only runs once.
        self._parse_all = memory.cache(self._parse_all_uncached)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(
        self,
        date_start: str,
        date_end: str,
        exclude_holdout: bool = True,
    ) -> pd.DataFrame:
        """Return long-format daily precipitation for the given date range.

        If ``exclude_holdout=True`` (default) and
        ``outputs/holdout_station_ids.json`` exists, the hold-out stations are
        removed so they are never seen by any model during training or CV.
        Pass ``exclude_holdout=False`` only for final hold-out evaluation.
        """
        all_data = self._parse_all(str(self._rekis_dir))
        mask = (all_data["date"] >= date_start) & (all_data["date"] <= date_end)
        result = all_data.loc[mask]

        # Filter to study area bounding box
        sa = self._cfg.study_area
        bbox_mask = (
            (result["lon"] >= sa.lon_min)
            & (result["lon"] <= sa.lon_max)
            & (result["lat"] >= sa.lat_min)
            & (result["lat"] <= sa.lat_max)
        )
        result = result.loc[bbox_mask].reset_index(drop=True)

        # Exclude hold-out stations from all training/CV pipelines
        if exclude_holdout:
            holdout_path = Path("outputs/holdout_station_ids.json")
            if holdout_path.exists():
                holdout_ids = set(json.loads(holdout_path.read_text()))
                result = result[~result["station_id"].isin(holdout_ids)].reset_index(drop=True)

        return result

    def load_stations(self) -> pd.DataFrame:
        """Return the station metadata table (one row per station)."""
        return self._read_stations(self._rekis_dir)

    # ------------------------------------------------------------------
    # Private: parsing (wrapped by joblib)
    # ------------------------------------------------------------------

    def _parse_all_uncached(self, rekis_dir: str) -> pd.DataFrame:
        """Read RR.csv, melt wide→long, join with station coordinates."""
        dir_path = Path(rekis_dir)
        precip_wide = self._read_precip_wide(dir_path)
        stations = self._read_stations(dir_path)
        return self._merge(precip_wide, stations)

    @staticmethod
    def _read_precip_wide(rekis_dir: Path) -> pd.DataFrame:
        """Read RR.csv (semicolon-separated, European decimal comma) → wide DataFrame."""
        df = pd.read_csv(
            rekis_dir / "RR.csv",
            sep=";",
            decimal=",",      # 3,8 → 3.8
            na_values=["-999", "-9999", ""],
            low_memory=False,
        )
        # 'zeit' looks like '1961-01-01T00:00:00Z' → '1961-01-01'
        df["zeit"] = pd.to_datetime(df["zeit"], utc=True).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"zeit": "date"})
        return df

    @staticmethod
    def _read_stations(rekis_dir: Path) -> pd.DataFrame:
        """Read Stationsliste.txt → tidy station table (deduplicated)."""
        df = pd.read_csv(rekis_dir / "Stationsliste.txt", sep=",")
        df = df.rename(columns={
            "Stat_ID": "station_id",
            "Laenge":  "lon",
            "Breite":  "lat",
            "Hoehe":   "elevation_m",
            "Name":    "station_name",
        })
        df = df.drop_duplicates(subset="station_id", keep="first")
        return df[["station_id", "lon", "lat", "elevation_m", "station_name"]]

    @staticmethod
    def _merge(wide: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
        """Melt wide→long, join coordinates, merge co-located gauges."""
        # wide: columns are ['date', 'DWD_2444', 'DWD_1960', ...]
        long = wide.melt(
            id_vars="date",
            var_name="station_id",
            value_name="precip_mm",
        )
        # Join coordinates
        long = long.merge(stations, on="station_id", how="left")

        # Merge co-located stations: map each ID to the first ID at that (lon, lat)
        canon = stations.groupby(["lon", "lat"])["station_id"].first()
        remap = stations.merge(
            canon.rename("_canon").reset_index(), on=["lon", "lat"],
        )
        n_merged = (remap["station_id"] != remap["_canon"]).sum()
        if n_merged > 0:
            id_map = dict(zip(remap["station_id"], remap["_canon"]))
            long["station_id"] = long["station_id"].map(id_map)
            long = long.groupby(["station_id", "date"], as_index=False).agg(
                lon=("lon", "first"),
                lat=("lat", "first"),
                elevation_m=("elevation_m", "first"),
                precip_mm=("precip_mm", "mean"),
            )

        # Sort for reproducibility
        long = long.sort_values(["date", "station_id"]).reset_index(drop=True)
        return long[["station_id", "date", "lon", "lat", "elevation_m", "precip_mm"]]
