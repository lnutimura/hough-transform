import cv2
import numpy as np
import pytest

from hough_transform.lines import (
    Line,
    detect_lines,
    find_lines,
    hough_lines,
    line_segments,
    sobel_magnitude,
)


def test_sobel_detects_both_edge_polarities():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 15:25] = 255

    gradient = sobel_magnitude(image)

    assert gradient[20, 14:16].max() > 0  # dark-to-bright edge
    assert gradient[20, 24:26].max() > 0  # bright-to-dark edge


def test_accumulator_counts_beyond_uint8_range():
    edges = np.zeros((400, 400), dtype=bool)
    edges[:, 100] = True

    accumulator, rhos, thetas = hough_lines(edges)

    assert accumulator[rhos == 100, thetas == 0].item() == 400


def test_finds_vertical_horizontal_and_diagonal_lines():
    edges = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(edges, (50, 0), (50, 199), 1)  # x = 50
    cv2.line(edges, (0, 30), (199, 30), 1)  # y = 30
    cv2.line(edges, (0, 150), (150, 0), 1)  # x + y = 150

    lines = find_lines(*hough_lines(edges.astype(bool)), min_votes=100)

    found = {(round(line.rho), round(np.rad2deg(line.theta))) for line in lines}
    assert found == {(50, 0), (-30, -90), (round(150 / np.sqrt(2)), 45)}


def test_segments_bridge_small_gaps_and_drop_short_runs():
    edges = np.zeros((50, 300), dtype=bool)
    edges[20, 10:60] = True
    edges[20, 65:120] = True  # 5 px gap: merged with the previous run
    edges[20, 140:200] = True  # 20 px gap: separate segment
    edges[20, 250:260] = True  # too short
    horizontal = Line(rho=-20, theta=-np.pi / 2, votes=0)

    segments = line_segments(edges, horizontal, min_length=30, max_gap=8)

    spans = sorted(tuple(sorted((s.start[0], s.end[0]))) for s in segments)
    assert spans == [(10, 119), (140, 199)]
    assert all(s.start[1] == s.end[1] == 20 for s in segments)


def test_detect_lines_on_a_drawn_grid():
    image = np.full((300, 300), 255, dtype=np.uint8)
    for position in (50, 150, 250):
        cv2.line(image, (position, 20), (position, 280), 0, 3)
        cv2.line(image, (20, position), (280, position), 0, 3)

    detection = detect_lines(image, min_votes=200, min_length=100)

    assert len(detection.lines) == 6
    assert len(detection.segments) == 6
    assert all(s.length == pytest.approx(260, abs=10) for s in detection.segments)
