from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class StudyArea(BaseModel):
    lon_min: float = 9.4
    lon_max: float = 16.0
    lat_min: float = 49.8
    lat_max: float = 53.8

    # ETRS89-LAEA — equal-area, metres, appropriate for central Europe
    target_crs: str = "EPSG:3035"
    grid_resolution_m: int = 1000  # 1 km


class KrigingParams(BaseModel):
    search_radius_km: float = 416.0   # p99 of pairwise inter-station distances (EDA notebook)
    n_stations_min: int = 3
    max_wet: int | None = 125         # max wet stations for local kriging (None = global)
    variogram_nlags: int = 38         # 416 km / 10.95 km ≈ 38 bins

    # Hofstra et al. 2008: spherical for indicator (occurrence), exponential for amounts.
    # "The spherical model was best for precipitation occurrence [sill at 470 km],
    #  the exponential model for precipitation amounts [sill at 1262 km]."
    variogram_model_indicator: str = "spherical"   # Stage 1: indicator kriging
    variogram_model_amount: str = "exponential"    # Stage 2: OK on normal scores

    # Hofstra 2008: probability threshold for wet-day classification = 0.4
    # (equals observed wet-day frequency, which is naturally < 0.5)
    indicator_probability_threshold: float = 0.4


class DataPaths(BaseModel):
    root: Path = Path("data")
    cache: Path = Path("data/cache")

    def ensure_dirs(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    study_area: StudyArea = StudyArea()
    kriging: KrigingParams = KrigingParams()
    paths: DataPaths = DataPaths()

    date_start: str = "1961-01-01"
    date_end: str = "2023-12-31"

    # Precipitation-specific preprocessing
    log_transform_offset: float = 0.001  # mm added before log to avoid log(0)
    wet_day_threshold_mm: float = 0.5   # below this → dry day; Haylock 2008: "threshold of 0.5 mm"

    random_seed: int = 42
