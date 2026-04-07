#!/usr/bin/env python3
"""
Core video loading, particle tracking, export, and stiffness helpers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np


Point = Tuple[float, float]
K_B = 1.380649e-23  # J/K
TrackingMode = str
CircleSeed = Tuple[float, float, float]


def rect_bounds(center: Point, radius: float, image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y = center
    height, width = image_shape
    return (
        max(0.0, x - radius),
        max(0.0, y - radius),
        min(float(width - 1), x + radius),
        min(float(height - 1), y + radius),
    )


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def iter_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float | None]:
    capture = cv2.VideoCapture(str(video_path))
    cv2_fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else None
    cv2_frames: List[np.ndarray] = []

    if capture.isOpened():
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                cv2_frames.append(frame)
        finally:
            capture.release()

    if cv2_frames:
        sample = cv2_frames[0]
        if np.max(sample) > 0:
            return cv2_frames, cv2_fps

    frames = list(iio.imiter(video_path))
    if not frames:
        raise RuntimeError("No frames were read from the video.")
    return frames, cv2_fps


def load_video_frames(
    video_path: Path,
    fps_override: float | None = None,
) -> Tuple[List[np.ndarray], float]:
    frames, detected_fps = iter_video_frames(video_path)
    fps = fps_override if fps_override is not None else detected_fps
    if not fps or fps <= 0:
        raise RuntimeError(
            "Could not determine FPS from the video. Please provide an FPS override."
        )
    return frames, float(fps)


def refine_centroid(image: np.ndarray, approx_point: Point, radius: int) -> Point:
    x0, y0 = approx_point
    height, width = image.shape

    x_min = max(int(round(x0)) - radius, 0)
    x_max = min(int(round(x0)) + radius + 1, width)
    y_min = max(int(round(y0)) - radius, 0)
    y_max = min(int(round(y0)) + radius + 1, height)

    roi = image[y_min:y_max, x_min:x_max].astype(np.float64)
    if roi.size == 0:
        return float(x0), float(y0)

    baseline = np.percentile(roi, 30)
    roi = np.clip(roi - baseline, a_min=0, a_max=None)
    total = roi.sum()
    if total <= 0:
        return float(x0), float(y0)

    yy, xx = np.indices(roi.shape)
    x_refined = x_min + float((xx * roi).sum() / total)
    y_refined = y_min + float((yy * roi).sum() / total)
    return x_refined, y_refined


def compute_gradient_feature(image: np.ndarray, blur_size: int) -> np.ndarray:
    blur_size = ensure_odd(max(3, blur_size))
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.GaussianBlur(magnitude, (blur_size, blur_size), 0)


def infer_tracking_mode(
    gray_frame: np.ndarray,
    center: Point,
    template_radius: int,
) -> TrackingMode:
    patch, _, _ = crop_square(gray_frame, center, template_radius)
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return "bright"

    patch = patch.astype(np.float32)
    height, width = patch.shape
    yy, xx = np.indices((height, width))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    center_mask = distances <= max(1.5, template_radius * 0.35)
    ring_mask = (distances >= max(2.0, template_radius * 0.45)) & (
        distances <= max(3.0, template_radius * 0.95)
    )
    if not np.any(center_mask) or not np.any(ring_mask):
        return "bright"

    center_mean = float(np.mean(patch[center_mask]))
    ring_mean = float(np.mean(patch[ring_mask]))
    contrast = ring_mean - center_mean
    return "ring" if contrast >= 4.0 else "bright"


def preprocess_tracking_image(
    image: np.ndarray,
    blur_size: int,
    mode: TrackingMode,
) -> np.ndarray:
    if mode == "ring":
        return compute_gradient_feature(image, blur_size)

    blur_size = ensure_odd(max(3, blur_size))
    return cv2.GaussianBlur(image, (blur_size, blur_size), 0).astype(np.float32)


def detect_particles(
    gray_frame: np.ndarray,
    num_particles: int,
    blur_size: int,
    threshold_percentile: float,
    min_area: float,
    search_radius: int,
) -> List[Point]:
    blur_size = ensure_odd(max(3, blur_size))
    blurred = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)

    threshold_value = np.percentile(blurred, threshold_percentile)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    candidates = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label]
        refined = refine_centroid(blurred, (cx, cy), search_radius)
        mean_intensity = float(blurred[labels == label].mean())
        candidates.append((mean_intensity, area, refined))

    if len(candidates) < num_particles:
        raise RuntimeError(
            f"Only found {len(candidates)} particle candidates; "
            f"expected {num_particles}. Adjust threshold/min-area if needed."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    points = [item[2] for item in candidates[:num_particles]]
    points.sort(key=lambda point: (point[1], point[0]))
    return points


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angular_difference_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def assign_points(
    previous_points: Sequence[Point],
    current_points: Sequence[Point],
    max_link_distance: float,
) -> List[Point]:
    if len(previous_points) != len(current_points):
        raise ValueError("Point count changed during assignment.")

    remaining = list(current_points)
    assigned: List[Point] = []

    for prev in previous_points:
        distances = [euclidean(prev, point) for point in remaining]
        best_index = int(np.argmin(distances))
        best_distance = distances[best_index]
        if best_distance > max_link_distance:
            raise RuntimeError(
                f"Particle moved {best_distance:.2f} px between frames, exceeding "
                f"the max-link-distance={max_link_distance:.2f} px."
            )
        assigned.append(remaining.pop(best_index))

    return assigned


def track_video_auto(
    video_path: Path,
    num_particles: int,
    fps_override: float | None,
    blur_size: int,
    threshold_percentile: float,
    min_area: float,
    max_link_distance: float,
    search_radius: int,
) -> Tuple[np.ndarray, float]:
    frames, fps = load_video_frames(video_path, fps_override=fps_override)

    tracks: List[List[Point]] = []
    previous_points: List[Point] | None = None

    for frame in frames:
        gray = to_grayscale(frame)
        detected = detect_particles(
            gray,
            num_particles=num_particles,
            blur_size=blur_size,
            threshold_percentile=threshold_percentile,
            min_area=min_area,
            search_radius=search_radius,
        )

        if previous_points is None:
            current_points = detected
        else:
            current_points = assign_points(
                previous_points, detected, max_link_distance=max_link_distance
            )

        tracks.append(current_points)
        previous_points = current_points

    return np.array(tracks, dtype=np.float64), fps


def crop_square(image: np.ndarray, center: Point, radius: int) -> Tuple[np.ndarray, int, int]:
    x, y = center
    height, width = image.shape
    x_min = max(int(round(x)) - radius, 0)
    x_max = min(int(round(x)) + radius + 1, width)
    y_min = max(int(round(y)) - radius, 0)
    y_max = min(int(round(y)) + radius + 1, height)
    return image[y_min:y_max, x_min:x_max], x_min, y_min


def fit_ring_circle(
    gray_frame: np.ndarray,
    previous_point: Point,
    expected_radius: float,
    roi_radius: int,
    blur_size: int,
    arc_angle_deg: float | None = None,
    arc_span_deg: float = 180.0,
) -> Tuple[Point, float, dict[str, float]]:
    search_roi, x_min, y_min = crop_square(gray_frame, previous_point, roi_radius)
    if search_roi.size == 0 or min(search_roi.shape[:2]) < 12:
        raise RuntimeError("Search window is too small for ring fitting.")

    blur_size = ensure_odd(max(3, blur_size))
    blurred = cv2.GaussianBlur(search_roi, (blur_size, blur_size), 0)

    min_radius = max(3, int(round(expected_radius * 0.55)))
    max_radius = max(min_radius + 2, int(round(expected_radius * 1.45)))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=max(8.0, expected_radius * 1.2),
        param1=80,
        param2=10,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        raise RuntimeError("No circular ring was detected in the local search region.")

    roi_center_x = previous_point[0] - x_min
    roi_center_y = previous_point[1] - y_min
    edge_map = cv2.Canny(blurred, 30, 90)
    edge_points = np.column_stack(np.nonzero(edge_map))
    best_score = None
    best_circle = None

    for circle in circles[0]:
        cx, cy, radius = map(float, circle)
        distance_penalty = math.hypot(cx - roi_center_x, cy - roi_center_y)
        radius_penalty = abs(radius - expected_radius) * 2.0
        arc_bonus = 0.0
        if edge_points.size > 0:
            ys = edge_points[:, 0].astype(np.float64)
            xs = edge_points[:, 1].astype(np.float64)
            distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            radial_mask = np.abs(distances - radius) <= max(1.5, radius * 0.25)
            if arc_angle_deg is not None:
                angles = np.degrees(np.arctan2(-(ys - cy), xs - cx))
                arc_mask = np.array(
                    [angular_difference_deg(angle, arc_angle_deg) <= arc_span_deg / 2.0 for angle in angles]
                )
                support = float(np.count_nonzero(radial_mask & arc_mask))
            else:
                support = float(np.count_nonzero(radial_mask))
            arc_bonus = min(20.0, support * 0.75)
        score = distance_penalty + radius_penalty - arc_bonus
        if best_score is None or score < best_score:
            best_score = score
            best_circle = (cx, cy, radius)

    if best_circle is None:
        raise RuntimeError("Ring fitting did not produce a usable circle.")

    cx_local, cy_local, fitted_radius = best_circle
    center = (x_min + cx_local, y_min + cy_local)
    local_patch, _, _ = crop_square(blurred, (cx_local, cy_local), max(4, int(fitted_radius)))
    contrast = float(local_patch.max() - np.percentile(local_patch, 20)) if local_patch.size else 0.0
    diagnostics = {
        "match_score": float(max(0.0, 1.0 - (best_score / max(roi_radius, 1)))),
        "contrast": contrast,
        "radius": fitted_radius,
        "arc_angle_deg": float(arc_angle_deg) if arc_angle_deg is not None else float("nan"),
        "previous_x": float(previous_point[0]),
        "previous_y": float(previous_point[1]),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "search_x0": float(x_min),
        "search_y0": float(y_min),
        "search_x1": float(x_min + search_roi.shape[1] - 1),
        "search_y1": float(y_min + search_roi.shape[0] - 1),
        "outline_radius": float(fitted_radius),
    }
    return center, fitted_radius, diagnostics


def extract_template(
    gray_frame: np.ndarray,
    center: Point,
    template_radius: int,
    blur_size: int,
) -> np.ndarray:
    patch, _, _ = crop_square(gray_frame, center, template_radius)
    if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
        raise RuntimeError("Selected particle is too close to the image border.")
    blur_size = ensure_odd(max(3, blur_size))
    return cv2.GaussianBlur(patch, (blur_size, blur_size), 0)


def localize_particle(
    gray_frame: np.ndarray,
    previous_point: Point,
    template: np.ndarray,
    tracking_mode: TrackingMode,
    roi_radius: int,
    blur_size: int,
    refine_radius: int,
    min_match_score: float,
    min_contrast: float,
) -> Tuple[Point, dict[str, float]]:
    search_roi, x_min, y_min = crop_square(gray_frame, previous_point, roi_radius)
    if (
        search_roi.shape[0] < template.shape[0]
        or search_roi.shape[1] < template.shape[1]
    ):
        raise RuntimeError("Search window became smaller than the tracking template.")

    search_processed = preprocess_tracking_image(search_roi, blur_size, tracking_mode)
    result = cv2.matchTemplate(search_processed, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < min_match_score:
        raise RuntimeError(
            f"Lost particle: template match score {max_val:.3f} "
            f"is below the threshold {min_match_score:.3f}."
        )

    top_left_x = max_loc[0]
    top_left_y = max_loc[1]
    center_x_local = top_left_x + (template.shape[1] - 1) / 2.0
    center_y_local = top_left_y + (template.shape[0] - 1) / 2.0
    refined_local = refine_centroid(
        search_processed, (center_x_local, center_y_local), refine_radius
    )

    local_patch, _, _ = crop_square(search_processed, refined_local, refine_radius)
    if local_patch.size == 0:
        raise RuntimeError("Lost particle: empty refinement region.")

    contrast = float(local_patch.max() - np.percentile(local_patch, 20))
    if contrast < min_contrast:
        raise RuntimeError(
            f"Lost particle: local contrast {contrast:.2f} is below "
            f"the threshold {min_contrast:.2f}."
        )

    refined = (refined_local[0] + x_min, refined_local[1] + y_min)
    outline_radius = float(max(template.shape[0], template.shape[1]) / 2.0)
    return refined, {
        "match_score": float(max_val),
        "contrast": contrast,
        "mode": 1.0 if tracking_mode == "ring" else 0.0,
        "previous_x": float(previous_point[0]),
        "previous_y": float(previous_point[1]),
        "center_x": float(refined[0]),
        "center_y": float(refined[1]),
        "search_x0": float(x_min),
        "search_y0": float(y_min),
        "search_x1": float(x_min + search_roi.shape[1] - 1),
        "search_y1": float(y_min + search_roi.shape[0] - 1),
        "outline_radius": outline_radius,
        "arc_angle_deg": float("nan"),
    }


def track_selected_particles(
    frames: Sequence[np.ndarray],
    seed_points: Sequence[Point],
    seed_radii: Sequence[float] | None = None,
    seed_arc_angles: Sequence[float | None] | None = None,
    roi_radius: int = 20,
    template_radius: int = 8,
    blur_size: int = 5,
    refine_radius: int = 6,
    min_match_score: float = 0.35,
    min_contrast: float = 10.0,
    tracking_mode: TrackingMode = "auto",
) -> Tuple[np.ndarray, list[list[dict[str, Any]]]]:
    if not seed_points:
        raise RuntimeError("No particles were selected for tracking.")
    if roi_radius <= template_radius:
        raise RuntimeError("roi_radius must be larger than template_radius.")

    gray_frames = [to_grayscale(frame) for frame in frames]
    if seed_radii is None:
        seed_radii = [float(template_radius)] * len(seed_points)
    if len(seed_radii) != len(seed_points):
        raise RuntimeError("seed_radii must match the number of selected particles.")
    if seed_arc_angles is None:
        seed_arc_angles = [None] * len(seed_points)
    if len(seed_arc_angles) != len(seed_points):
        raise RuntimeError("seed_arc_angles must match the number of selected particles.")
    resolved_modes: list[TrackingMode] = []
    templates: list[np.ndarray] = []
    for point, radius in zip(seed_points, seed_radii):
        mode = tracking_mode
        if mode == "auto":
            mode = infer_tracking_mode(gray_frames[0], point, max(3, int(round(radius))))
        resolved_modes.append(mode)
        raw_template = extract_template(
            gray_frames[0], point, max(3, int(round(radius))), blur_size
        )
        templates.append(preprocess_tracking_image(raw_template, blur_size, mode))

    tracks: List[List[Point]] = [[(float(x), float(y)) for x, y in seed_points]]
    frame_shape = gray_frames[0].shape
    diagnostics: list[list[dict[str, Any]]] = [[]]
    for index, point in enumerate(seed_points):
        radius = float(seed_radii[index])
        arc_angle = seed_arc_angles[index]
        x0, y0, x1, y1 = rect_bounds(point, roi_radius, frame_shape)
        diagnostics[0].append(
            {
                "match_score": 1.0,
                "contrast": float("nan"),
                "mode": 1.0 if resolved_modes[index] == "ring" else 0.0,
                "mode_name": resolved_modes[index],
                "previous_x": float(point[0]),
                "previous_y": float(point[1]),
                "center_x": float(point[0]),
                "center_y": float(point[1]),
                "search_x0": x0,
                "search_y0": y0,
                "search_x1": x1,
                "search_y1": y1,
                "outline_radius": radius,
                "radius": radius,
                "arc_angle_deg": float(arc_angle) if arc_angle is not None else float("nan"),
                "selection_mode": "arc" if arc_angle is not None else "full",
            }
        )

    previous_points = [(float(x), float(y)) for x, y in seed_points]
    current_radii = [float(radius) for radius in seed_radii]
    current_arc_angles = list(seed_arc_angles)

    for frame_index, gray in enumerate(gray_frames[1:], start=1):
        current_points: List[Point] = []
        frame_diag: list[dict[str, float]] = []
        next_radii: list[float] = []

        for particle_index, prev_point in enumerate(previous_points):
            try:
                if resolved_modes[particle_index] == "ring":
                    refined, fitted_radius, diag = fit_ring_circle(
                        gray_frame=gray,
                        previous_point=prev_point,
                        expected_radius=current_radii[particle_index],
                        roi_radius=max(roi_radius, int(round(current_radii[particle_index] * 2.2))),
                        blur_size=blur_size,
                        arc_angle_deg=current_arc_angles[particle_index],
                    )
                    diag["mode"] = 1.0
                    diag["mode_name"] = "ring"
                    diag["selection_mode"] = (
                        "arc" if current_arc_angles[particle_index] is not None else "full"
                    )
                    next_radii.append(fitted_radius)
                else:
                    refined, diag = localize_particle(
                        gray_frame=gray,
                        previous_point=prev_point,
                        template=templates[particle_index],
                        tracking_mode=resolved_modes[particle_index],
                        roi_radius=roi_radius,
                        blur_size=blur_size,
                        refine_radius=refine_radius,
                        min_match_score=min_match_score,
                        min_contrast=min_contrast,
                    )
                    diag["mode_name"] = resolved_modes[particle_index]
                    diag["selection_mode"] = "full"
                    diag["radius"] = current_radii[particle_index]
                    next_radii.append(current_radii[particle_index])
                diag["radius"] = next_radii[-1]
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Tracking failed for particle {particle_index + 1} "
                    f"at frame {frame_index}: {exc}"
                ) from exc

            current_points.append(refined)
            frame_diag.append(diag)

        tracks.append(current_points)
        diagnostics.append(frame_diag)
        previous_points = current_points
        current_radii = next_radii
        current_arc_angles = list(seed_arc_angles)

    return np.array(tracks, dtype=np.float64), diagnostics


def build_output_table(
    tracks: np.ndarray,
    fps: float,
    pixel_size: float,
) -> Tuple[np.ndarray, List[str]]:
    frame_count, num_particles, _ = tracks.shape
    time_s = np.arange(frame_count, dtype=np.float64) / fps
    scaled_positions = tracks * pixel_size
    mean_positions = scaled_positions.mean(axis=0)
    displacements = scaled_positions - mean_positions

    columns = [np.arange(frame_count, dtype=np.int64), time_s]
    headers = ["frame", "time_s"]

    for particle_index in range(num_particles):
        x = scaled_positions[:, particle_index, 0]
        y = scaled_positions[:, particle_index, 1]
        dx = displacements[:, particle_index, 0]
        dy = displacements[:, particle_index, 1]

        columns.extend([x, y, dx, dy])
        particle_id = particle_index + 1
        headers.extend(
            [
                f"p{particle_id}_x",
                f"p{particle_id}_y",
                f"p{particle_id}_dx",
                f"p{particle_id}_dy",
            ]
        )

    table = np.column_stack(columns)
    return table, headers


def write_dat(
    output_path: Path,
    table: np.ndarray,
    headers: Sequence[str],
    unit: str,
    pixel_size: float,
    fps: float,
    video_path: Path,
    temperature_k: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = [
        f"# source_video = {video_path}",
        f"# fps = {fps:.8f}",
        f"# coordinate_unit = {unit}",
        f"# pixel_size = {pixel_size:.8f} {unit}/pixel",
    ]
    if temperature_k is not None:
        metadata.append(f"# temperature_K = {temperature_k:.8f}")
    metadata.extend(
        [
            "# dx and dy are displacements relative to each particle's time-averaged position",
            "# " + "\t".join(headers),
        ]
    )

    np.savetxt(
        output_path,
        table,
        fmt=["%d", "%.8f"] + ["%.8f"] * (table.shape[1] - 2),
        delimiter="\t",
        header="\n".join(metadata),
        comments="",
    )


def load_headers(dat_file: Path) -> list[str]:
    for line in dat_file.read_text().splitlines():
        if line.startswith("# frame"):
            return line[2:].split("\t")
    raise RuntimeError("Could not find header line in .dat file.")


def estimate_stiffness(
    table: np.ndarray,
    headers: Sequence[str],
    pixel_size_um: float,
    temperature_k: float,
) -> list[dict[str, Any]]:
    pixel_size_m = pixel_size_um * 1e-6
    kbt = K_B * temperature_k
    results: list[dict[str, Any]] = []

    for idx, name in enumerate(headers):
        if not (name.endswith("_dx") or name.endswith("_dy")):
            continue

        values = table[:, idx]
        variance_units2 = float(np.var(values, ddof=1))
        variance_m2 = variance_units2 * (pixel_size_m ** 2)
        variance_um2 = variance_units2 * (pixel_size_um ** 2)
        if variance_m2 <= 0:
            k_n_per_m = float("nan")
            k_pn_per_um = float("nan")
        else:
            k_n_per_m = kbt / variance_m2
            k_pn_per_um = k_n_per_m * 1e6

        results.append(
            {
                "column": name,
                "variance_px2": variance_units2,
                "variance_um2": variance_um2,
                "k_n_per_m": k_n_per_m,
                "k_pn_per_um": k_pn_per_um,
                "k_pn_per_nm": k_pn_per_um / 1000.0,
            }
        )

    return results


def load_dat_table(dat_file: Path) -> Tuple[np.ndarray, list[str]]:
    headers = load_headers(dat_file)
    table = np.loadtxt(dat_file, comments="#")
    return table, headers
