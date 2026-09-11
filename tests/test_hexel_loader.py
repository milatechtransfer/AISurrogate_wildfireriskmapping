from unittest.mock import MagicMock

import numpy as np

from data_preparation import hexel_loader


def _masked_grid() -> np.ma.MaskedArray:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return np.ma.masked_array(np.ones((4, 4), dtype=np.float32), mask=mask)


def test_crop_window_is_created_only_for_unmasked_data(tmp_path, monkeypatch):
    profile = {"crs": "EPSG:3978", "transform": None, "width": 4, "height": 4}
    monkeypatch.setattr(hexel_loader, "load_spatial_raster", lambda *args, **kwargs: (_masked_grid(), profile))
    monkeypatch.setattr(hexel_loader, "load_fuel_grid", lambda *args, **kwargs: _masked_grid())
    monkeypatch.setattr(hexel_loader, "load_ignition_grid", lambda *args, **kwargs: _masked_grid())
    save_crop_window = MagicMock()
    monkeypatch.setattr(hexel_loader, "save_crop_window", save_crop_window)

    feature_map_path = tmp_path / "feature_channel_map_1.json"
    masked_features, _, _ = hexel_loader.load_spatial_features_per_hexel(
        root_dir=str(tmp_path),
        hex_id="01",
        feature_channel_map_path=str(feature_map_path),
        mask_scope="actual",
        ignition_weighting="max",
    )

    assert masked_features is not None
    assert masked_features.shape[1:3] == (4, 4)
    save_crop_window.assert_not_called()

    unmasked_features, _, _ = hexel_loader.load_spatial_features_per_hexel(
        root_dir=str(tmp_path),
        hex_id="01",
        feature_channel_map_path=str(feature_map_path),
        mask_scope=None,
        ignition_weighting="max",
    )

    assert unmasked_features is not None
    assert unmasked_features.shape[1:3] == (3, 3)
    save_crop_window.assert_called_once_with(
        str(feature_map_path),
        hex_id="01",
        mask_scope=None,
        window=(1, 1, 3, 3),
    )
