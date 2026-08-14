from pathlib import Path

import pytest
import yaml

from src.config_io import _load_raw_config


def test_config_inheritance_deep_merges_nested_values(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(yaml.safe_dump({"model": {"architecture": "auto", "hidden_features": [8, 16]}, "training": {"max_epochs": 5}}))
    child.write_text(yaml.safe_dump({"extends": "base.yaml", "model": {"architecture": "mechanistic_propagation"}}))

    resolved = _load_raw_config(child)

    assert resolved["model"] == {"architecture": "mechanistic_propagation", "hidden_features": [8, 16]}
    assert resolved["training"]["max_epochs"] == 5


def test_config_inheritance_rejects_cycles(tmp_path: Path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")

    with pytest.raises(ValueError, match="cycle"):
        _load_raw_config(first)
