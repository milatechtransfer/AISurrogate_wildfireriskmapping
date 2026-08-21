from pathlib import Path

import pandas as pd
import pytest

from src.datasets.fuel_utils import _FUEL_CURVES_CSV, build_fuel_curve_lookup


@pytest.mark.parametrize(
    ("feature_name", "expected"),
    [
        ("iROS", [1.0, 2.0]),
        ("HFI", [10.0, 20.0]),
        ("iROS_HFI", [1.0, 2.0, 10.0, 20.0]),
    ],
)
def test_build_fuel_curve_lookup_reads_combined_curve_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_name: str,
    expected: list[float],
) -> None:
    pd.DataFrame(
        {
            "fbp_code": [1, 1],
            "SeasonState": ["direct", "direct"],
            "ISI": [0.0, 5.0],
            "ROS": [1.0, 2.0],
            "HFI": [10.0, 20.0],
        }
    ).to_csv(tmp_path / _FUEL_CURVES_CSV, index=False)
    monkeypatch.setattr("src.datasets.fuel_utils.find_hex_ids", lambda _root: [])

    lookup = build_fuel_curve_lookup(
        root_dir=tmp_path,
        raw_data_dir=tmp_path,
        feature_name=feature_name,
    )

    assert lookup[(1, None)].tolist() == expected
