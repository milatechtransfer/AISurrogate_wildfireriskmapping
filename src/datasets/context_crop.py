"""Utilities for centered supervision crops within larger context patches."""

from __future__ import annotations

from typing import Any


def centered_crop_slices(
    input_height: int,
    input_width: int,
    crop_height: int,
    crop_width: int,
) -> tuple[slice, slice]:
    """Return spatial slices for an exactly centered crop."""
    if min(input_height, input_width, crop_height, crop_width) <= 0:
        raise ValueError("Input and crop dimensions must be positive.")
    if crop_height > input_height or crop_width > input_width:
        raise ValueError(f"Crop shape {(crop_height, crop_width)} cannot exceed input shape {(input_height, input_width)}.")

    height_difference = input_height - crop_height
    width_difference = input_width - crop_width
    if height_difference % 2 != 0 or width_difference % 2 != 0:
        raise ValueError(
            f"Centered crop requires even input-crop differences, got input {(input_height, input_width)} "
            f"and crop {(crop_height, crop_width)}."
        )

    row_start = height_difference // 2
    col_start = width_difference // 2
    return (
        slice(row_start, row_start + crop_height),
        slice(col_start, col_start + crop_width),
    )


def configured_target_crop(config: dict[str, Any] | None) -> tuple[int, int] | None:
    """Read an explicitly configured target crop from a checkpoint config."""
    data_prep = (config or {}).get("data_prep", {})
    crop_height = data_prep.get("target_crop_h")
    crop_width = data_prep.get("target_crop_w")
    if crop_height is None and crop_width is None:
        return None
    if crop_height is None or crop_width is None:
        raise ValueError("Checkpoint data_prep must set both target_crop_h and target_crop_w.")
    return int(crop_height), int(crop_width)


def validate_context_crop_metadata(
    metadata: Any,
    configured_crop: tuple[int, int] | None,
) -> None:
    """Ensure prepared patch geometry matches the model's crop configuration."""
    required_columns = {"input_win_h", "input_win_w", "target_crop_h", "target_crop_w"}
    columns = set(metadata.columns)
    if not required_columns <= columns:
        if configured_crop is not None:
            raise ValueError("Context-crop config requires patch metadata with input and target crop dimensions.")
        return

    geometry = metadata[list(sorted(required_columns))].drop_duplicates()
    if len(geometry) != 1:
        raise ValueError("Patch metadata must contain one consistent input-window and target-crop geometry.")

    row = geometry.iloc[0]
    input_shape = (int(row["input_win_h"]), int(row["input_win_w"]))
    metadata_crop = (int(row["target_crop_h"]), int(row["target_crop_w"]))
    metadata_uses_crop = metadata_crop != input_shape
    if configured_crop is None:
        if metadata_uses_crop:
            raise ValueError(
                f"Prepared patches use target crop {metadata_crop} within input window {input_shape}, "
                "but the model config has no context crop."
            )
        return
    if configured_crop != metadata_crop:
        raise ValueError(f"Configured target crop {configured_crop} does not match prepared patch metadata {metadata_crop}.")
