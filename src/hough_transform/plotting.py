import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from hough_transform.circles import CircleDetection
from hough_transform.lines import LineDetection

RED = (230, 40, 40)
BLACK = (0, 0, 0)


def _show(ax, image: np.ndarray, title: str, **kwargs) -> None:
    ax.imshow(image, **kwargs)
    ax.set_title(title)
    ax.set_axis_off()


def plot_lines(rgb: np.ndarray, detection: LineDetection) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    _show(axes[0, 0], rgb, "Input")
    _show(axes[0, 1], detection.gradient, "Sobel gradient magnitude", cmap="gray")
    _show(axes[0, 2], detection.edges, "Edges (thresholded gradient)", cmap="gray")

    ax = axes[1, 0]
    degrees = np.rad2deg(detection.thetas)
    extent = (degrees[0], degrees[-1], detection.rhos[0], detection.rhos[-1])
    ax.imshow(detection.accumulator, cmap="magma", extent=extent, aspect="auto", origin="lower")
    ax.scatter(
        [np.rad2deg(line.theta) for line in detection.lines],
        [line.rho for line in detection.lines],
        s=40,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1,
    )
    ax.set(
        title=f"Hough space ({len(detection.lines)} peaks)",
        xlabel="θ (degrees)",
        ylabel="ρ (pixels)",
    )

    height, width = detection.edges.shape
    diagonal = int(np.hypot(height, width))
    infinite = rgb.copy()
    for line in detection.lines:
        cos, sin = np.cos(line.theta), np.sin(line.theta)
        x0, y0 = line.rho * cos, line.rho * sin
        p1 = (round(x0 + diagonal * sin), round(y0 - diagonal * cos))
        p2 = (round(x0 - diagonal * sin), round(y0 + diagonal * cos))
        cv2.line(infinite, p1, p2, RED, 1, cv2.LINE_AA)
    _show(axes[1, 1], infinite, "Detected lines")

    segments = rgb.copy()
    for segment in detection.segments:
        cv2.line(segments, segment.start, segment.end, RED, 2, cv2.LINE_AA)
    _show(axes[1, 2], segments, f"Detected segments ({len(detection.segments)})")
    return fig


def plot_circles(rgb: np.ndarray, detection: CircleDetection) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    _show(axes[0, 0], rgb, "Input")
    _show(axes[0, 1], detection.edges, "Canny edges", cmap="gray")

    radii = f"r = {detection.radii[0]}–{detection.radii[-1]} px"
    _show(
        axes[1, 0],
        detection.accumulator.max(axis=0),
        f"Hough space, max over {radii}",
        cmap="magma",
    )

    output = rgb.copy()
    for circle in detection.circles:
        center = (circle.x, circle.y)
        cv2.circle(output, center, circle.radius, BLACK, 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, BLACK, -1, cv2.LINE_AA)
    _show(axes[1, 1], output, f"Detected circles ({len(detection.circles)})")
    return fig
