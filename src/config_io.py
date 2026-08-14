"""Configuration loading with optional YAML inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import Config


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in ancestors:
        cycle = " -> ".join(str(item) for item in (*ancestors, path))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_raw_config(parent_path, (*ancestors, path)), raw)


def load_config(path: str | Path) -> Config:
    return Config(**_load_raw_config(Path(path)))
