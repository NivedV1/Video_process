#!/usr/bin/env python3
"""
Track bright particles in a video and export x/y motion to a .dat file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from particle_tracking_core import build_output_table, track_video_auto, write_dat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track bright particles in a video and export motion as .dat."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("particle_motion.dat"),
        help="Output .dat file path (default: particle_motion.dat).",
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=2,
        help="Number of particles to track (default: 2).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS. If omitted, FPS is read from the file.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=1.0,
        help="Physical size of one pixel in your preferred unit (default: 1.0).",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="pixel",
        help="Unit label for pixel-size scaling, e.g. um or nm (default: pixel).",
    )
    parser.add_argument(
        "--blur",
        type=int,
        default=5,
        help="Gaussian blur kernel size, odd integer (default: 5).",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile used to isolate bright particles (default: 99.0).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=8.0,
        help="Minimum blob area in pixels for a valid particle (default: 8).",
    )
    parser.add_argument(
        "--max-link-distance",
        type=float,
        default=30.0,
        help="Maximum movement between consecutive frames in pixels (default: 30).",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=12,
        help="ROI radius used for subpixel centroid refinement (default: 12).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks, fps = track_video_auto(
        video_path=args.video,
        num_particles=args.num_particles,
        fps_override=args.fps,
        blur_size=args.blur,
        threshold_percentile=args.threshold_percentile,
        min_area=args.min_area,
        max_link_distance=args.max_link_distance,
        search_radius=args.search_radius,
    )
    table, headers = build_output_table(
        tracks=tracks,
        fps=fps,
        pixel_size=args.pixel_size,
    )
    write_dat(
        output_path=args.output,
        table=table,
        headers=headers,
        unit=args.unit,
        pixel_size=args.pixel_size,
        fps=fps,
        video_path=args.video,
    )
    print(f"Tracked {tracks.shape[1]} particles across {tracks.shape[0]} frames.")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
