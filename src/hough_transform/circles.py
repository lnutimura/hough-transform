"""Circle detection with the (x, y, r) Hough Transform."""

from dataclasses import dataclass

import cv2
import numpy as np

from hough_transform.peaks import find_peaks


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int
    score: float


@dataclass
class CircleDetection:
    """Every intermediate stage of the pipeline, kept around for plotting."""

    edges: np.ndarray
    accumulator: np.ndarray
    radii: np.ndarray
    circles: list[Circle]


def circle_offsets(radius: int) -> np.ndarray:
    """Distinct integer ``(dy, dx)`` offsets of the pixels on a circle of ``radius``."""
    angles = np.linspace(0, 2 * np.pi, int(np.ceil(16 * np.pi * radius)), endpoint=False)
    offsets = np.round(radius * np.stack([np.sin(angles), np.cos(angles)], axis=1))
    return np.unique(offsets.astype(int), axis=0)


def hough_circles(edges: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Vote every edge pixel into an (r, y, x) accumulator.

    Each edge pixel votes for all the centers of the circles of each radius passing through
    it. Votes are normalized by the circle perimeter, so ``accumulator[i, y, x]`` is the
    fraction of the circle of radius ``radii[i]`` centered at ``(x, y)`` that is covered by
    edges, regardless of the radius.
    """
    height, width = edges.shape
    ys, xs = np.nonzero(edges)
    accumulator = np.zeros((radii.size, height, width), dtype=np.float32)
    for i, radius in enumerate(radii):
        offsets = circle_offsets(int(radius))
        cy = (ys[:, None] - offsets[:, 0]).ravel()
        cx = (xs[:, None] - offsets[:, 1]).ravel()
        inside = (cy >= 0) & (cy < height) & (cx >= 0) & (cx < width)
        votes = np.bincount(cy[inside] * width + cx[inside], minlength=height * width)
        accumulator[i] = votes.reshape(height, width) / len(offsets)
    return accumulator


def find_circles(
    accumulator: np.ndarray, radii: np.ndarray, threshold: float, min_distance: int
) -> list[Circle]:
    """Pick circles whose perimeter is at least ``threshold`` covered by edges.

    Centers closer than ``min_distance`` pixels are merged, keeping the best radius and
    position among them.
    """
    best_score = accumulator.max(axis=0)
    best_radius = accumulator.argmax(axis=0)
    window = (2 * min_distance + 1, 2 * min_distance + 1)
    return [
        Circle(int(x), int(y), int(radii[best_radius[y, x]]), float(best_score[y, x]))
        for y, x in find_peaks(best_score, window, threshold)
    ]


def detect_circles(
    gray: np.ndarray,
    min_radius: int = 40,
    max_radius: int = 80,
    threshold: float = 0.5,
    min_distance: int = 20,
) -> CircleDetection:
    """Full pipeline: Canny edges, Hough voting and peak picking."""
    edges = cv2.Canny(gray, 100, 200) > 0
    radii = np.arange(min_radius, max_radius + 1)
    accumulator = hough_circles(edges, radii)
    circles = find_circles(accumulator, radii, threshold, min_distance)
    return CircleDetection(edges, accumulator, radii, circles)
