import numpy as np
from scipy import ndimage


def find_peaks(accumulator: np.ndarray, window: tuple[int, ...], threshold: float) -> np.ndarray:
    """Return the indices of the strongest, well-separated cells of a Hough accumulator.

    A cell is a peak if its value is at least ``threshold`` and no stronger peak lies
    within ``window`` (the full window size along each axis). Peaks are returned
    strongest first, as an array of shape ``(n_peaks, accumulator.ndim)``.
    """
    is_local_max = ndimage.maximum_filter(accumulator, size=window, mode="constant") == accumulator
    candidates = np.argwhere(is_local_max & (accumulator >= threshold))
    candidates = candidates[np.argsort(-accumulator[tuple(candidates.T)], kind="stable")]

    # Plateaus produce several equal local maxima next to each other; keep only one.
    half = np.array(window) // 2
    peaks: list[np.ndarray] = []
    for candidate in candidates:
        if all(np.any(np.abs(candidate - peak) > half) for peak in peaks):
            peaks.append(candidate)
    return np.array(peaks, dtype=int).reshape(-1, accumulator.ndim)
