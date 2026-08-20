from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.process_ignition_count import (
    NORMALIZED_CV_COLUMN,
    NORMALIZED_LOG_MEAN_COLUMN,
    RAW_CV_COLUMN,
    RAW_LOG_MEAN_COLUMN,
    add_normalized_count_columns,
    ignition_count_statistics,
    read_ignition_count_pmf,
)


def test_user_defined_ignition_count_distribution(tmp_path: Path):
    count_path = tmp_path / "count.csv"
    count_path.write_text("Mean,DistributionType\n,ignition_count\n")
    scenario_frame = pd.DataFrame(
        {
            "Name": ["ignition_count", "ignition_count"],
            "Value": [1, 5],
            "RelativeFrequency": [3, 1],
        }
    )

    pmf = read_ignition_count_pmf(count_path, scenario_frame)
    statistics = ignition_count_statistics(pmf)

    assert pmf == pytest.approx({1.0: 0.75, 5.0: 0.25})
    assert statistics["IGNITION_COUNT_MEAN"] == pytest.approx(2.0)
    assert statistics["IGNITION_COUNT_SD"] == pytest.approx(np.sqrt(3.0))
    assert statistics[RAW_CV_COLUMN] == pytest.approx(np.sqrt(3.0) / 2.0)


def test_count_normalization_uses_unique_training_hexes_without_clipping(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "hex_id": [1, 1, 2, 3],
            "GRIDCODE": [10, 11, 20, 30],
            RAW_LOG_MEAN_COLUMN: [1.0, 1.0, 3.0, 5.0],
            RAW_CV_COLUMN: [0.5, 0.5, 1.0, 1.5],
        }
    )

    normalized, ranges = add_normalized_count_columns(
        frame,
        train_hex_ids={1, 2},
        norm_params_path=tmp_path / "count_norm.json",
    )

    assert ranges == ((1.0, 3.0), (0.5, 1.0))
    assert normalized.loc[0, NORMALIZED_LOG_MEAN_COLUMN] == 0.0
    assert normalized.loc[2, NORMALIZED_LOG_MEAN_COLUMN] == 1.0
    assert normalized.loc[3, NORMALIZED_LOG_MEAN_COLUMN] == 2.0
    assert normalized.loc[3, NORMALIZED_CV_COLUMN] == 2.0
