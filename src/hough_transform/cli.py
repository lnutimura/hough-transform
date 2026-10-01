import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from hough_transform.circles import detect_circles
from hough_transform.lines import detect_lines
from hough_transform.plotting import plot_circles, plot_lines


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hough", description="Detect lines or circles with a from-scratch Hough Transform."
    )
    subparsers = parser.add_subparsers(dest="shape", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("image", type=Path, help="input image")
    common.add_argument(
        "-o", "--output", type=Path, help="save the figure here instead of opening a window"
    )

    lines = subparsers.add_parser("lines", parents=[common], help="detect line segments")
    lines.add_argument(
        "--edge-threshold",
        type=float,
        default=300.0,
        help="minimum Sobel gradient magnitude of an edge pixel (default: 300)",
    )
    lines.add_argument(
        "--min-votes", type=int, default=200, help="minimum edge pixels on a line (default: 200)"
    )
    lines.add_argument(
        "--min-length", type=int, default=50, help="minimum segment length in pixels (default: 50)"
    )
    lines.add_argument(
        "--max-gap",
        type=int,
        default=8,
        help="maximum gap in pixels bridged within a segment (default: 8)",
    )

    circles = subparsers.add_parser("circles", parents=[common], help="detect circles")
    circles.add_argument("--min-radius", type=int, default=40, help="(default: 40)")
    circles.add_argument("--max-radius", type=int, default=80, help="(default: 80)")
    circles.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="minimum fraction of the perimeter covered by edges (default: 0.5)",
    )
    circles.add_argument(
        "--min-distance",
        type=int,
        default=20,
        help="minimum distance in pixels between centers (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"error: cannot read image {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if args.shape == "lines":
        detection = detect_lines(
            gray, args.edge_threshold, args.min_votes, args.min_length, args.max_gap
        )
        print("segment  start (x, y)  end (x, y)  length")
        for i, s in enumerate(detection.segments):
            print(f"{i:>7}  {str(s.start):>12}  {str(s.end):>10}  {s.length:>6.1f}")
        fig = plot_lines(rgb, detection)
    else:
        detection = detect_circles(
            gray, args.min_radius, args.max_radius, args.threshold, args.min_distance
        )
        print("circle  center (x, y)  radius  score")
        for i, c in enumerate(detection.circles):
            print(f"{i:>6}  {str((c.x, c.y)):>13}  {c.radius:>6}  {c.score:.2f}")
        fig = plot_circles(rgb, detection)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=100)
    else:
        plt.show()
