import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual import fire_size_counterfactual_transform as fs
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig


def _scenario(**overrides: float) -> ScenarioConfig:
    params: dict[str, object] = {
        "mode": "spread_day_quantile_scaling",
        "spread_day_delta_q50_days": 0.4,
        "spread_day_delta_q90_days": 5.0,
        "size_scaling_exponent": 2.0,
    }
    params.update(overrides)
    return ScenarioConfig(
        name="spread_day_fire_size",
        kind="fire_size",
        description="",
        params=params,
    )


def _write_inputs(tmp_path: Path, *, include_s2: bool = True) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    pd.DataFrame(
        {
            "GRIDCODE": [10] * 5 + [99] * 5,
            "NORM_LOG_SIZE_HA": [0.0, 0.25, 0.5, 0.75, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    ).to_csv(data_root / "df_fire_fru_processed.csv", index=False)
    (data_root / "fire_size_norm_params.json").write_text(json.dumps({"log_size_min": 0.0, "log_size_max": 2.0}))

    tabular = tmp_path / "raw" / "hex16" / "tabular"
    tabular.mkdir(parents=True)
    pd.DataFrame(
        {
            "Season": ["s1", "s1", "s2"],
            "Cause": ["Human", "Lightning", "Human"],
            "FireZone": ["fru10", "fru10", "fru10"],
            "RelativeLikelihood": [1.0, 1.0, 2.0],
        }
    ).to_csv(tabular / "hex16_IgnitionDistribution.csv", index=False)
    spread_rows = [
        {
            "Name": "Spread Event Distribution - fru10 - s1",
            "Value": 1,
            "RelativeFrequency": 100.0,
        }
    ]
    if include_s2:
        spread_rows.append(
            {
                "Name": "Spread Event Distribution - fru10 - s2",
                "Value": 4,
                "RelativeFrequency": 100.0,
            }
        )
    pd.DataFrame(spread_rows).to_csv(
        tabular / "hex16_ScenarioDistributions - B - FINAL.csv",
        index=False,
    )
    return data_root, tmp_path / "raw"


def test_ignition_season_weights_aggregates_causes_then_normalizes() -> None:
    frame = pd.DataFrame(
        {
            "FireZone": ["fru01", "fru01", "fru01", "fru02"],
            "Season": ["s1", "s1", "s2", "s2"],
            "Cause": ["Human", "Lightning", "Human", "Human"],
            "RelativeLikelihood": [2.0, 1.0, 3.0, 4.0],
        }
    )

    weights = fs.ignition_season_weights(frame)

    assert weights == {1: {"s1": 0.5, "s2": 0.5}, 2: {"s2": 1.0}}


def test_mixture_quantiles_uses_mixture_cdf_not_weighted_component_quantiles() -> None:
    quantiles = fs.mixture_quantiles(
        {
            "s1": (np.asarray([1]), np.asarray([1.0])),
            "s2": (np.asarray([10]), np.asarray([1.0])),
        },
        {"s1": 0.6, "s2": 0.4},
        (0.5, 0.9),
    )

    assert quantiles == {0.5: 1.0, 0.9: 10.0}


def test_spread_pmfs_renormalizes_small_rounding_drift_but_rejects_large_error() -> None:
    frame = pd.DataFrame(
        {
            "Name": ["Spread Event Distribution - fru10 - s1"] * 2,
            "Value": [1, 2],
            "RelativeFrequency": [50.0, 50.6],
        }
    )

    pmfs = fs._spread_pmfs(frame, source="spread.csv", total_tolerance_percent=1.0)

    assert pmfs[(10, "s1")].probabilities.sum() == pytest.approx(1.0)
    assert pmfs[(10, "s1")].original_total_percent == pytest.approx(100.6)

    frame["RelativeFrequency"] = [40.0, 40.0]
    with pytest.raises(ValueError, match=r"sums to 80.*outside"):
        fs._spread_pmfs(frame, source="spread.csv", total_tolerance_percent=1.0)


def test_spread_pmfs_rejects_malformed_schema() -> None:
    frame = pd.DataFrame(
        {
            "wrongName": ["Spread Event Distribution - fru10 - s1"],
            "Value": [1],
            "RelativeFrequency": [100.0],
        }
    )

    with pytest.raises(ValueError, match=r"missing required column.*Name"):
        fs._spread_pmfs(frame, source="spread.csv", total_tolerance_percent=1.0)


def test_materialize_fire_size_scenario_writes_exact_hex_scoped_q3_lookup(tmp_path: Path) -> None:
    data_root, raw_root = _write_inputs(tmp_path)
    prediction_dir = tmp_path / "predictions" / "scenario" / "bp"

    result = fs.materialize_fire_size_scenario(
        scenario=_scenario(),
        raw_data_dir=raw_root,
        processed_fire_size_csv=data_root / "df_fire_fru_processed.csv",
        recipient_hex_ids=["16"],
        prediction_dir=prediction_dir,
        feature_name="NORM_LOG_SIZE_HA",
        zone_id_col="GRIDCODE",
        quantiles=[0.1, 0.5, 0.9],
    )

    lookup = pd.read_csv(result.edited_csv_path)
    zone = lookup.loc[lookup["GRIDCODE"].eq(10)].iloc[0]
    unchanged = lookup.loc[lookup["GRIDCODE"].eq(99)].iloc[0]
    assert result.feature_columns == (
        "NORM_LOG_SIZE_HA_q10",
        "NORM_LOG_SIZE_HA_q50",
        "NORM_LOG_SIZE_HA_q90",
    )
    assert zone["hex_id"] == 16
    assert zone["NORM_LOG_SIZE_HA_q10"] == pytest.approx(0.1)
    assert zone["NORM_LOG_SIZE_HA_q50"] > 0.5
    assert zone["NORM_LOG_SIZE_HA_q90"] > 0.9
    assert unchanged[["NORM_LOG_SIZE_HA_q10", "NORM_LOG_SIZE_HA_q50", "NORM_LOG_SIZE_HA_q90"]].to_numpy(dtype=float) == pytest.approx(
        [0.24, 0.4, 0.56]
    )

    summary = result.summary.iloc[0]
    assert summary["spread_day_q50"] == pytest.approx(1.0)
    assert summary["spread_day_q90"] == pytest.approx(4.0)
    assert summary["fire_size_multiplier_q50"] == pytest.approx(1.96)
    assert summary["fire_size_multiplier_q90"] == pytest.approx(5.0625)
    assert summary["future_fire_size_q10_ha"] == pytest.approx(summary["baseline_fire_size_q10_ha"])
    assert json.loads(summary["season_weights"]) == {"s1": 0.5, "s2": 0.5}

    fill = pd.read_csv(result.global_fill_csv_path)
    assert fill["hex_id"].tolist() == [16]
    assert list(fill.columns) == ["hex_id", *result.feature_columns]


def test_materialize_fire_size_scenario_rejects_missing_weighted_season(tmp_path: Path) -> None:
    data_root, raw_root = _write_inputs(tmp_path, include_s2=False)

    with pytest.raises(ValueError, match=r"missing spread distributions.*s2"):
        fs.materialize_fire_size_scenario(
            scenario=_scenario(),
            raw_data_dir=raw_root,
            processed_fire_size_csv=data_root / "df_fire_fru_processed.csv",
            recipient_hex_ids=["16"],
            prediction_dir=tmp_path / "predictions",
            feature_name="NORM_LOG_SIZE_HA",
            zone_id_col="GRIDCODE",
            quantiles=[0.1, 0.5, 0.9],
        )


def test_materialize_fire_size_scenario_rejects_quantile_crossing(tmp_path: Path) -> None:
    data_root, raw_root = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match=r"violates q10 <= q50 <= q90"):
        fs.materialize_fire_size_scenario(
            scenario=_scenario(
                spread_day_delta_q50_days=100.0,
                spread_day_delta_q90_days=0.0,
            ),
            raw_data_dir=raw_root,
            processed_fire_size_csv=data_root / "df_fire_fru_processed.csv",
            recipient_hex_ids=["16"],
            prediction_dir=tmp_path / "predictions",
            feature_name="NORM_LOG_SIZE_HA",
            zone_id_col="GRIDCODE",
            quantiles=[0.1, 0.5, 0.9],
        )


def test_materialize_fire_size_scenario_supports_negative_deltas(tmp_path: Path) -> None:
    """Shortening spread events is a valid intervention: deltas are signed."""
    data_root, raw_root = _write_inputs(tmp_path)

    result = fs.materialize_fire_size_scenario(
        scenario=_scenario(spread_day_delta_q50_days=0.0, spread_day_delta_q90_days=-2.0),
        raw_data_dir=raw_root,
        processed_fire_size_csv=data_root / "df_fire_fru_processed.csv",
        recipient_hex_ids=["16"],
        prediction_dir=tmp_path / "predictions" / "scenario" / "bp",
        feature_name="NORM_LOG_SIZE_HA",
        zone_id_col="GRIDCODE",
        quantiles=[0.1, 0.5, 0.9],
    )

    summary = result.summary.iloc[0]
    # spread q90 = 4 days -> 2 days, beta=2 => (2/4) ** 2 = 0.25.
    assert summary["fire_size_multiplier_q50"] == pytest.approx(1.0)
    assert summary["fire_size_multiplier_q90"] == pytest.approx(0.25)
    assert summary["future_fire_size_q90_ha"] < summary["baseline_fire_size_q90_ha"]
    assert summary["future_fire_size_q50_ha"] == pytest.approx(summary["baseline_fire_size_q50_ha"])


def test_materialize_fire_size_scenario_rejects_non_positive_future_spread_days(tmp_path: Path) -> None:
    data_root, raw_root = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="non-positive"):
        fs.materialize_fire_size_scenario(
            scenario=_scenario(spread_day_delta_q50_days=-5.0),
            raw_data_dir=raw_root,
            processed_fire_size_csv=data_root / "df_fire_fru_processed.csv",
            recipient_hex_ids=["16"],
            prediction_dir=tmp_path / "predictions",
            feature_name="NORM_LOG_SIZE_HA",
            zone_id_col="GRIDCODE",
            quantiles=[0.1, 0.5, 0.9],
        )
