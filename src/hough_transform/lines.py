"""Line detection with the (rho, theta) Hough Transform.

A line is parametrized by its normal form ``rho = x cos(theta) + y sin(theta)``, where
``x`` is the column, ``y`` is the row, ``theta`` is in [-90, 90) degrees and ``rho`` is
the signed distance from the image origin (top-left corner).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from hough_transform.peaks import find_peaks

SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T

Point = tuple[int, int]


@dataclass(frozen=True)
class Line:
    rho: float
    theta: float
    votes: int


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point

    @property
    def length(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))


@dataclass
class LineDetection:
    """Every intermediate stage of the pipeline, kept around for plotting."""

    gradient: np.ndarray
    edges: np.ndarray
    accumulator: np.ndarray
    rhos: np.ndarray
    thetas: np.ndarray
    lines: list[Line]
    segments: list[Segment]


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a grayscale image, computed with 3x3 Sobel kernels."""
    gray = gray.astype(np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, SOBEL_X)
    gy = cv2.filter2D(gray, cv2.CV_32F, SOBEL_Y)
    return np.hypot(gx, gy)


def hough_lines(
    edges: np.ndarray, theta_step: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vote every edge pixel into a (rho, theta) accumulator.

    Returns ``(accumulator, rhos, thetas)`` where ``accumulator[i, j]`` counts the edge
    pixels lying on the line ``(rhos[i], thetas[j])``. Rho has a 1-pixel resolution and
    theta (in radians) a ``theta_step``-degree resolution.
    """
    height, width = edges.shape
    diagonal = int(np.ceil(np.hypot(height, width)))
    rhos = np.arange(-diagonal, diagonal + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step))

    ys, xs = np.nonzero(edges)
    accumulator = np.zeros((rhos.size, thetas.size), dtype=np.int64)
    for j, theta in enumerate(thetas):
        rho_indexes = np.round(xs * np.cos(theta) + ys * np.sin(theta)).astype(int) + diagonal
        accumulator[:, j] = np.bincount(rho_indexes, minlength=rhos.size)
    return accumulator, rhos, thetas


def find_lines(
    accumulator: np.ndarray,
    rhos: np.ndarray,
    thetas: np.ndarray,
    min_votes: int,
    window: tuple[int, int] = (15, 5),
) -> list[Line]:
    """Pick the accumulator peaks with at least ``min_votes`` votes.

    ``window`` is the (rho, theta) neighborhood, in accumulator cells, within which
    only the strongest line is kept.
    """
    peaks = find_peaks(accumulator, window, min_votes)
    return [Line(float(rhos[i]), float(thetas[j]), int(accumulator[i, j])) for i, j in peaks]


def _sample_line(line: Line, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Integer pixel coordinates, 1 pixel apart, of the part of ``line`` inside the image."""
    height, width = shape
    cos, sin = np.cos(line.theta), np.sin(line.theta)
    diagonal = np.hypot(height, width)
    t = np.arange(-diagonal, diagonal)
    xs = np.round(line.rho * cos - t * sin).astype(int)
    ys = np.round(line.rho * sin + t * cos).astype(int)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    return xs[inside], ys[inside]


def line_segments(edges: np.ndarray, line: Line, min_length: int, max_gap: int) -> list[Segment]:
    """Split an infinite Hough line into the finite segments actually backed by edges.

    Walks along the line and groups the edge pixels it crosses into runs, allowing gaps of
    up to ``max_gap`` pixels; runs shorter than ``min_length`` pixels are discarded.
    """
    xs, ys = _sample_line(line, edges.shape)
    on_edge = np.flatnonzero(edges[ys, xs])
    if on_edge.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(on_edge) > max_gap + 1)
    starts = on_edge[np.r_[0, breaks + 1]]
    ends = on_edge[np.r_[breaks, on_edge.size - 1]]
    return [
        Segment((int(xs[s]), int(ys[s])), (int(xs[e]), int(ys[e])))
        for s, e in zip(starts, ends, strict=True)
        if e - s >= min_length
    ]


def detect_lines(
    gray: np.ndarray,
    edge_threshold: float = 300.0,
    min_votes: int = 200,
    min_length: int = 50,
    max_gap: int = 8,
) -> LineDetection:
    """Full pipeline: Sobel edges, Hough voting, peak picking and segment extraction."""
    gradient = sobel_magnitude(gray)
    edges = gradient > edge_threshold
    accumulator, rhos, thetas = hough_lines(edges)
    lines = find_lines(accumulator, rhos, thetas, min_votes)

    # Thicken the edges so that segments survive the rounding of the line samples.
    thick_edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    segments = [s for line in lines for s in line_segments(thick_edges, line, min_length, max_gap)]
    return LineDetection(gradient, edges, accumulator, rhos, thetas, lines, segments)
