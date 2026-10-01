import cv2
import numpy as np

from hough_transform.circles import circle_offsets, detect_circles


def test_circle_offsets_lie_on_the_circle():
    offsets = circle_offsets(20)

    distances = np.hypot(offsets[:, 0], offsets[:, 1])
    assert np.all(np.abs(distances - 20) <= np.sqrt(2) / 2)
    assert len(np.unique(offsets, axis=0)) == len(offsets)


def test_detects_circles_of_different_radii():
    image = np.full((200, 300), 255, dtype=np.uint8)
    expected = [(60, 70, 25), (160, 100, 40), (250, 60, 30)]
    for x, y, radius in expected:
        cv2.circle(image, (x, y), radius, 90, -1)

    detection = detect_circles(image, min_radius=15, max_radius=50)

    found = sorted((c.x, c.y, c.radius) for c in detection.circles)
    assert len(found) == len(expected)
    for (x, y, radius), (fx, fy, fradius) in zip(expected, found, strict=True):
        assert abs(fx - x) <= 1 and abs(fy - y) <= 1 and abs(fradius - radius) <= 1


def test_ignores_shapes_that_are_not_circles():
    image = np.full((200, 200), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), 0, -1)

    assert detect_circles(image, min_radius=20, max_radius=60).circles == []
