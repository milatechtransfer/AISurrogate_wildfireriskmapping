import numpy as np


def stitch_windows(
    windows: list[np.ndarray], coords: list[tuple], masks: list[np.ndarray], original_shape: tuple, mode: str = "mean"
) -> np.ndarray:
    """
    Reconstructs an image from overlapping windows using either averaging or maximization.

    Args:
        windows (list of np.array): List of window arrays (H_win, W_win, C).
        coords (list of tuples): List of (row, col) top-left coordinates for each window.
        masks (list of np.ndarray): List of masks for the windows. True if valid value
        original_shape (tuple): Shape of the target hexel (H, W, C).
        mode (str): How to combine/stitch the windows (Options: mean, max)
    Returns:
        np.array: The reconstructed image (Shape: original_shape, (H,W))
    """
    dtype = np.float32

    if mode == "mean":
        # --- AVERAGE MODE ---
        accumulator = np.zeros(original_shape, dtype=dtype)
        counter = np.zeros(original_shape, dtype=dtype)

        for window, mask, (r, c) in zip(windows, masks, coords, strict=False):
            window = np.array(window, copy=True)
            window[~mask] = 0.0
            h_win, w_win = window.shape[:2]

            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Accumulate Sum and Count
            accumulator[r:r_end, c:c_end] += window[:h_paste, :w_paste]
            counter[r:r_end, c:c_end] += mask[:h_paste, :w_paste]

        # Normalize
        valid_mask = counter > 0
        reconstructed = np.full(original_shape, np.nan, dtype=dtype)
        reconstructed[valid_mask] = accumulator[valid_mask] / counter[valid_mask]

        return reconstructed

    elif mode == "max":
        # --- MAX MODE ---
        # Initialize with negative infinity so any real data (even negative) will override it
        accumulator = np.full(original_shape, -np.inf, dtype=dtype)

        for window, mask, (r, c) in zip(windows, masks, coords, strict=False):
            window = np.array(window, copy=True)
            h_win, w_win = window.shape[:2]
            window[~mask] = -np.inf
            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Update the area with the element-wise maximum
            current_area = accumulator[r:r_end, c:c_end]
            new_data = window[:h_paste, :w_paste]

            accumulator[r:r_end, c:c_end] = np.maximum(current_area, new_data)

        # Leave areas with no valid window contribution as NaN
        accumulator[np.isinf(accumulator)] = np.nan

        return accumulator

    else:
        raise ValueError(f"Unknown mode: {mode}")
