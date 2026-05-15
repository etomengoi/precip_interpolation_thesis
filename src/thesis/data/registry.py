"""DataRegistry — composition root wiring all data sources."""
from dataclasses import dataclass, field

from thesis.config import Config
from thesis.data.dem import DEMSource
from thesis.data.rekis import ReKISSource
from thesis.data.soilgrids import SoilGridsSource


@dataclass
class DataRegistry:
    stations: ReKISSource
    dem: DEMSource
    soilgrids: dict[str, SoilGridsSource] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Config) -> "DataRegistry":
        """Instantiate all loaders from a single Config object."""
        return cls(
            stations=ReKISSource(cfg),
            dem=DEMSource(cfg),
            soilgrids={
                var: SoilGridsSource(cfg, variable=var)
                for var in ("bulk_density", "clay", "sand", "silt", "soc", "water_10kpa")
            },
        )
