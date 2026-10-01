import cv2
import matplotlib
import numpy as np
import pytest

from hough_transform.cli import main

matplotlib.use("Agg")


def test_circles_command_saves_figure_and_prints_detections(tmp_path, capsys):
    image = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.circle(image, (75, 75), 40, (200, 50, 50), -1)
    cv2.imwrite(str(tmp_path / "input.png"), image)
    output = tmp_path / "out" / "result.png"

    main(["circles", str(tmp_path / "input.png"), "-o", str(output), "--min-radius", "30"])

    assert output.exists()
    assert "(75, 75)" in capsys.readouterr().out


def test_lines_command_saves_figure(tmp_path):
    image = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.line(image, (75, 10), (75, 140), (0, 0, 0), 3)
    cv2.imwrite(str(tmp_path / "input.png"), image)

    main(
        [
            "lines",
            str(tmp_path / "input.png"),
            "-o",
            str(tmp_path / "result.png"),
            "--min-votes",
            "100",
        ]
    )

    assert (tmp_path / "result.png").exists()


def test_unreadable_image_exits_with_error(tmp_path):
    with pytest.raises(SystemExit, match="cannot read image"):
        main(["lines", str(tmp_path / "missing.png")])
