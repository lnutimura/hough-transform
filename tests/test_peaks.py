import numpy as np

from hough_transform.peaks import find_peaks


def test_returns_separated_peaks_strongest_first():
    accumulator = np.zeros((20, 20))
    accumulator[5, 5] = 3
    accumulator[15, 15] = 7
    accumulator[5, 6] = 2  # next to a stronger peak

    peaks = find_peaks(accumulator, window=(3, 3), threshold=1)

    assert peaks.tolist() == [[15, 15], [5, 5]]


def test_plateau_yields_a_single_peak():
    accumulator = np.zeros((10, 10))
    accumulator[4:6, 4:6] = 5

    assert len(find_peaks(accumulator, window=(3, 3), threshold=1)) == 1


def test_no_peaks_above_threshold():
    assert find_peaks(np.ones((5, 5)), window=(3, 3), threshold=2).shape == (0, 2)
