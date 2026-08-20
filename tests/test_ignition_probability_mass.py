import numpy as np
import pandas as pd
import pytest

from data_preparation.spatial.ignition import build_ignition_probability_mass_channels


def test_probability_mass_matches_category_then_location_sampling():
    grids = {
        ("H", "s1"): np.ma.array([[1.0, 3.0], [7.0, 9.0]]),
        ("N", "s1"): np.ma.array([[8.0, 6.0], [2.0, 2.0]]),
    }
    distribution = pd.DataFrame(
        {
            "Season": ["s1", "s1"],
            "Cause": ["Human", "Lightning"],
            "FireZone": ["zone_a", "zone_b"],
            "RelativeLikelihood": [3.0, 1.0],
        }
    )
    firezones = np.ma.array([[1, 1], [2, 2]])

    channels = build_ignition_probability_mass_channels(
        grids=grids,
        distribution_frame=distribution,
        firezones_grid=firezones,
        zone_name_to_id={"zone_a": 1, "zone_b": 2},
        scale=100.0,
    )

    assert channels.shape == (2, 2, 2)
    assert channels[..., 0].sum() == pytest.approx(75.0)
    assert channels[..., 1].sum() == pytest.approx(25.0)
    assert channels[0, 0, 0] == pytest.approx(18.75)
    assert channels[0, 1, 0] == pytest.approx(56.25)
    assert channels[1, 0, 1] == pytest.approx(12.5)
    assert channels[1, 1, 1] == pytest.approx(12.5)
    assert channels.sum() == pytest.approx(100.0)
