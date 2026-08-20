from pathlib import Path
from typing import Literal

# Mask scopes select which pixels are scored at evaluation:
#   "actual"      - the inner hex only
#   "buffer"      - the full buffered extent: inner hex + surrounding ring
#   "buffer_only" - the surrounding ring only, with the inner hex excluded
MaskScope = Literal["actual", "buffer", "buffer_only"]
MASK_SCOPE_CHOICES: tuple[MaskScope, ...] = ("actual", "buffer", "buffer_only")


def normalize_mask_scope(mask_scope: str) -> MaskScope:
    if mask_scope == "actual":
        return "actual"
    if mask_scope == "buffer":
        return "buffer"
    if mask_scope == "buffer_only":
        return "buffer_only"
    raise ValueError(f"mask_scope must be one of {MASK_SCOPE_CHOICES}, got {mask_scope!r}.")


def prepared_mask_scope(mask_scope: str) -> MaskScope:
    """Return the patch-data scope a requested scope is computed from.

    "buffer_only" is backed by "buffer" patches because the ring only exists in
    buffer-extent data; the inner hex is excluded later, at evaluation. "actual"
    and "buffer" are backed by patches of the same scope.
    """
    scope = normalize_mask_scope(mask_scope)
    if scope == "buffer_only":
        return "buffer"
    return scope


class Paths:
    def __init__(self, hex_id: str, root_dir: str | Path = ".") -> None:
        self.base_dir = Path(root_dir) / f"hex{hex_id}"
        self.spatial_dir = self.base_dir / "spatial"
        self.tabular_dir = self.base_dir / "tabular"
        self.hex_id = hex_id

    def ignition_prob_dir(self) -> Path:
        return self.spatial_dir / "ignition_grids"

    def mask_grid_actual(self, hex_id: int | str) -> Path:
        return self.spatial_dir / "mask_grids" / f"hex{hex_id}_actual.shp"

    def mask_grid_buffer(self, hex_id: int | str) -> Path:
        return self.spatial_dir / "mask_grids" / f"hex{hex_id}_buffer.shp"

    def mask_grid(self, hex_id: int | str, mask_scope: str = "actual") -> Path:
        scope = normalize_mask_scope(mask_scope)
        if scope == "actual":
            return self.mask_grid_actual(hex_id)
        return self.mask_grid_buffer(hex_id)

    def firezones_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_firezones.tif"

    def fuel_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_fbp.tif"

    def elevation_grid(self, hex_id: int | str) -> Path:
        return self.spatial_dir / f"hex{hex_id}_dem.tif"

    def fuel_table(self, hex_id: int | str | None = None) -> Path:
        if not hex_id:
            hex_id = self.hex_id
        return self.tabular_dir / f"hex{hex_id}_FuelTypes.csv"

    def weather_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_DailyWeather.csv"

    def ignition_distribution_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_IgnitionDistribution.csv"

    def ignition_count_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_IgnitionCount.csv"

    def spread_event_days_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_SpreadEventDays.csv"

    def daily_burning_hours_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_DailyBurningHours.csv"

    def scenario_distributions_table(self, hex_id: int | str) -> Path:
        matches = sorted(self.tabular_dir.glob(f"hex{hex_id}_ScenarioDistributions*FINAL.csv"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one scenario-distribution table for hex{hex_id}, found {len(matches)}: {matches}")
        return matches[0]

    def seasons_greenup_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_GreenUp.csv"

    def firezones_table(self, hex_id: int | str) -> Path:
        return self.tabular_dir / f"hex{hex_id}_FireZones.csv"

    def output_burn_prob(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputBurnProbability" / "burnProbability-sn2.tif"

    def output_fire_intensity(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputFireIntensitySummaryMap" / "fbpSummary-FireIntensity-Average.tif"

    def output_ros(self) -> Path:
        return self.base_dir / "results" / "burnP3Plus_OutputRateOfSpreadSummaryMap" / "fbpSummary-RateOfSpread-Average.tif"
