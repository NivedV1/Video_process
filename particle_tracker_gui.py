#!/usr/bin/env python3
"""
Semi-manual GUI for particle tracking and trap stiffness estimation.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from particle_tracking_core import (
    build_output_table,
    estimate_stiffness,
    load_video_frames,
    track_selected_particles,
    write_dat,
)


MAX_DISPLAY_SIZE = 900


class ParticleTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Particle Tracker for Trap Stiffness")

        self.video_path: Path | None = None
        self.frames: list[np.ndarray] = []
        self.fps: float | None = None
        self.seed_points: list[tuple[float, float]] = []
        self.seed_radii: list[float] = []
        self.seed_arc_angles: list[float | None] = []
        self.tracks: np.ndarray | None = None
        self.diagnostics: list[list[dict[str, float | str]]] | None = None
        self.output_table: np.ndarray | None = None
        self.output_headers: list[str] | None = None
        self.stiffness_results: list[dict[str, float]] = []
        self.display_scale = 1.0
        self.tk_image: tk.PhotoImage | None = None
        self.drag_start: tuple[float, float] | None = None
        self.preview_circle_id: int | None = None
        self.play_job: str | None = None

        self.video_var = tk.StringVar()
        self.pixel_size_var = tk.StringVar(value="1.0")
        self.temperature_var = tk.StringVar(value="25.0")
        self.roi_radius_var = tk.StringVar(value="20")
        self.template_radius_var = tk.StringVar(value="8")
        self.match_threshold_var = tk.StringVar(value="0.35")
        self.tracking_mode_var = tk.StringVar(value="Auto")
        self.selection_mode_var = tk.StringVar(value="Full circle")
        self.status_var = tk.StringVar(value="Load a video to begin.")
        self.show_search_var = tk.BooleanVar(value=True)
        self.show_outline_var = tk.BooleanVar(value=True)
        self.show_center_var = tk.BooleanVar(value=True)
        self.show_path_var = tk.BooleanVar(value=True)
        self.frame_index_var = tk.IntVar(value=0)
        self.frame_label_var = tk.StringVar(value="Frame 0 / 0")
        self.frame_status_var = tk.StringVar(
            value="Tracking diagnostics will appear here after processing."
        )

        self._build_ui()
        self._update_buttons()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Video").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.video_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 6)
        )
        ttk.Button(controls, text="Browse", command=self.browse_video).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Button(controls, text="Load", command=self.load_video).grid(
            row=0, column=3, sticky="ew", padx=(6, 0)
        )

        ttk.Label(controls, text="Pixel size (um/px)").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.pixel_size_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(6, 6)
        )
        ttk.Label(controls, text="Temperature (C)").grid(row=1, column=2, sticky="e")
        ttk.Entry(controls, textvariable=self.temperature_var, width=12).grid(
            row=1, column=3, sticky="w", padx=(6, 0)
        )

        ttk.Label(controls, text="Search radius (px)").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.roi_radius_var, width=12).grid(
            row=2, column=1, sticky="w", padx=(6, 6)
        )
        ttk.Label(controls, text="Template radius (px)").grid(row=2, column=2, sticky="e")
        ttk.Entry(controls, textvariable=self.template_radius_var, width=12).grid(
            row=2, column=3, sticky="w", padx=(6, 0)
        )

        ttk.Label(controls, text="Match threshold").grid(row=3, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.match_threshold_var, width=12).grid(
            row=3, column=1, sticky="w", padx=(6, 6)
        )
        ttk.Label(controls, text="Particle type").grid(row=3, column=2, sticky="e")
        ttk.Combobox(
            controls,
            textvariable=self.tracking_mode_var,
            values=("Auto", "Bright spot", "Ring / hollow"),
            state="readonly",
            width=14,
        ).grid(row=3, column=3, sticky="w", padx=(6, 0))
        ttk.Label(controls, text="Selection").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.selection_mode_var,
            values=("Full circle", "Half circle / arc"),
            state="readonly",
            width=14,
        ).grid(row=4, column=1, sticky="w", padx=(6, 6))

        action_bar = ttk.Frame(controls)
        action_bar.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        action_bar.columnconfigure(5, weight=1)

        self.undo_button = ttk.Button(action_bar, text="Undo Last", command=self.undo_last)
        self.undo_button.grid(row=0, column=0, padx=(0, 6))
        self.clear_button = ttk.Button(action_bar, text="Clear All", command=self.clear_points)
        self.clear_button.grid(row=0, column=1, padx=(0, 6))
        self.track_button = ttk.Button(action_bar, text="Start Tracking", command=self.start_tracking)
        self.track_button.grid(row=0, column=2, padx=(0, 6))
        self.save_button = ttk.Button(action_bar, text="Save .dat", command=self.save_dat)
        self.save_button.grid(row=0, column=3, padx=(0, 6))
        ttk.Label(action_bar, textvariable=self.status_var).grid(row=0, column=5, sticky="e")

        content = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        content.grid(row=1, column=0, sticky="nsew")

        viewer_frame = ttk.Frame(content, padding=(10, 0, 10, 10))
        viewer_frame.columnconfigure(0, weight=1)
        viewer_frame.rowconfigure(0, weight=1)
        content.add(viewer_frame, weight=3)

        self.canvas = tk.Canvas(viewer_frame, background="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        diagnostics_controls = ttk.LabelFrame(viewer_frame, text="Tracking Diagnostics", padding=8)
        diagnostics_controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        diagnostics_controls.columnconfigure(7, weight=1)

        ttk.Checkbutton(
            diagnostics_controls,
            text="Show search area",
            variable=self.show_search_var,
            command=self._refresh_canvas,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            diagnostics_controls,
            text="Show found outline",
            variable=self.show_outline_var,
            command=self._refresh_canvas,
        ).grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            diagnostics_controls,
            text="Show center",
            variable=self.show_center_var,
            command=self._refresh_canvas,
        ).grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            diagnostics_controls,
            text="Show path",
            variable=self.show_path_var,
            command=self._refresh_canvas,
        ).grid(row=0, column=3, sticky="w", padx=(0, 8))

        self.prev_button = ttk.Button(
            diagnostics_controls, text="Prev", command=lambda: self._step_frame(-1)
        )
        self.prev_button.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.play_button = ttk.Button(
            diagnostics_controls, text="Play", command=self._toggle_playback
        )
        self.play_button.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.next_button = ttk.Button(
            diagnostics_controls, text="Next", command=lambda: self._step_frame(1)
        )
        self.next_button.grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Label(diagnostics_controls, textvariable=self.frame_label_var).grid(
            row=1, column=3, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        self.frame_slider = ttk.Scale(
            diagnostics_controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_slider_change,
        )
        self.frame_slider.grid(row=1, column=4, columnspan=4, sticky="ew", padx=(12, 0), pady=(8, 0))

        ttk.Label(
            diagnostics_controls,
            textvariable=self.frame_status_var,
            justify="left",
        ).grid(row=2, column=0, columnspan=8, sticky="ew", pady=(8, 0))

        results_frame = ttk.Frame(content, padding=(0, 0, 10, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        content.add(results_frame, weight=2)

        ttk.Label(
            results_frame,
            text="Results",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.results_text = tk.Text(results_frame, width=45, height=28, state="disabled")
        self.results_text.grid(row=1, column=0, sticky="nsew")

    def _update_buttons(self) -> None:
        loaded = bool(self.frames)
        has_points = bool(self.seed_points)
        has_output = self.output_table is not None and self.output_headers is not None
        has_processed = self.tracks is not None and self.diagnostics is not None
        self.undo_button.configure(state="normal" if has_points else "disabled")
        self.clear_button.configure(state="normal" if has_points else "disabled")
        self.track_button.configure(state="normal" if loaded and has_points else "disabled")
        self.save_button.configure(state="normal" if has_output else "disabled")
        playback_state = "normal" if has_processed else "disabled"
        self.prev_button.configure(state=playback_state)
        self.next_button.configure(state=playback_state)
        self.play_button.configure(state=playback_state)
        if not has_processed:
            self.play_button.configure(text="Play")

    def browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.mpg *.mpeg"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.video_var.set(selected)

    def load_video(self) -> None:
        video_text = self.video_var.get().strip()
        if not video_text:
            messagebox.showerror("Missing video", "Choose a video file first.")
            return

        video_path = Path(video_text)
        if not video_path.exists():
            messagebox.showerror("Missing video", f"File not found:\n{video_path}")
            return

        try:
            frames, fps = load_video_frames(video_path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.video_path = video_path
        self.frames = frames
        self.fps = fps
        self.seed_points = []
        self.seed_radii = []
        self.seed_arc_angles = []
        self.tracks = None
        self.diagnostics = None
        self.output_table = None
        self.output_headers = None
        self.stiffness_results = []
        self._stop_playback()
        self.frame_index_var.set(0)
        self.frame_slider.configure(to=max(0, len(frames) - 1), value=0)
        self.frame_label_var.set(f"Frame 1 / {len(frames)}")
        self.frame_status_var.set(
            "Draw particles first. After tracking, this panel will show frame-by-frame diagnostics."
        )
        self._display_frame(frames[0])
        self._set_results_text(
            f"Loaded {video_path.name}\n"
            f"Frames: {len(frames)}\n"
            f"FPS: {fps:.3f}\n\n"
            "Draw one circle around each particle in the first frame.\n"
            "For overlapping rings, switch Selection to Half circle / arc and drag toward the clean side of the ring."
        )
        self.status_var.set("Video loaded. Draw circles around particles in order.")
        self._update_buttons()

    def _frame_to_photoimage(self, frame: np.ndarray) -> tk.PhotoImage:
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width = rgb.shape[:2]
        scale = min(MAX_DISPLAY_SIZE / max(width, 1), MAX_DISPLAY_SIZE / max(height, 1), 1.0)
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        if scale != 1.0:
            rgb = cv2.resize(rgb, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        self.display_scale = scale
        ppm_header = f"P6 {scaled_width} {scaled_height} 255 ".encode("ascii")
        ppm_data = ppm_header + rgb.astype(np.uint8).tobytes()
        return tk.PhotoImage(data=ppm_data, format="PPM")

    def _display_frame(self, frame: np.ndarray, draw_seeds: bool = True) -> None:
        self.tk_image = self._frame_to_photoimage(frame)
        self.canvas.configure(
            width=self.tk_image.width(),
            height=self.tk_image.height(),
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        if draw_seeds:
            self._draw_seed_markers()

    def _particle_color(self, particle_index: int) -> str:
        colors = [
            "#ffcc00",
            "#33d1ff",
            "#ff6b6b",
            "#7eff99",
            "#c299ff",
            "#ffa94d",
        ]
        return colors[particle_index % len(colors)]

    def _draw_seed_markers(self) -> None:
        for idx, ((x, y), radius_px, arc_angle) in enumerate(
            zip(self.seed_points, self.seed_radii, self.seed_arc_angles), start=1
        ):
            cx = x * self.display_scale
            cy = y * self.display_scale
            radius = max(6.0, radius_px * self.display_scale)
            if arc_angle is None:
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    outline="#00ff99",
                    width=2,
                )
            else:
                start = arc_angle - 90.0
                self.canvas.create_arc(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    start=start,
                    extent=180.0,
                    style=tk.ARC,
                    outline="#00ff99",
                    width=2,
                )
            self.canvas.create_line(cx - 4, cy, cx + 4, cy, fill="#00ff99", width=2)
            self.canvas.create_line(cx, cy - 4, cx, cy + 4, fill="#00ff99", width=2)
            self.canvas.create_text(
                cx + 10,
                cy - 10,
                text=f"p{idx}",
                anchor="sw",
                fill="#00ff99",
                font=("Segoe UI", 10, "bold"),
            )

    def _refresh_canvas(self) -> None:
        if not self.frames:
            return
        if self.tracks is None or self.diagnostics is None:
            self._display_frame(self.frames[0])
            return
        self._render_processed_frame(self.frame_index_var.get())

    def _render_processed_frame(self, frame_index: int) -> None:
        if not self.frames or self.tracks is None or self.diagnostics is None:
            return

        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        self.frame_index_var.set(frame_index)
        if int(round(float(self.frame_slider.get()))) != frame_index:
            self.frame_slider.set(frame_index)

        self._display_frame(self.frames[frame_index], draw_seeds=False)
        self._draw_diagnostics(frame_index)
        self.frame_label_var.set(f"Frame {frame_index + 1} / {len(self.frames)}")
        self.frame_status_var.set(self._format_frame_status(frame_index))

    def _draw_diagnostics(self, frame_index: int) -> None:
        if self.tracks is None or self.diagnostics is None:
            return

        for particle_index in range(self.tracks.shape[1]):
            color = self._particle_color(particle_index)
            diag = self.diagnostics[frame_index][particle_index]
            center = self.tracks[frame_index, particle_index]
            cx = float(center[0]) * self.display_scale
            cy = float(center[1]) * self.display_scale

            if self.show_path_var.get():
                points = []
                for path_point in self.tracks[: frame_index + 1, particle_index]:
                    points.extend(
                        [
                            float(path_point[0]) * self.display_scale,
                            float(path_point[1]) * self.display_scale,
                        ]
                    )
                if len(points) >= 4:
                    self.canvas.create_line(*points, fill=color, width=2, smooth=True)

            if self.show_search_var.get():
                self.canvas.create_rectangle(
                    float(diag["search_x0"]) * self.display_scale,
                    float(diag["search_y0"]) * self.display_scale,
                    float(diag["search_x1"]) * self.display_scale,
                    float(diag["search_y1"]) * self.display_scale,
                    outline=color,
                    dash=(5, 3),
                    width=2,
                )

            if self.show_outline_var.get():
                radius = float(diag["outline_radius"]) * self.display_scale
                self.canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    outline=color,
                    width=2,
                )
                arc_angle = float(diag["arc_angle_deg"])
                if not math.isnan(arc_angle):
                    self.canvas.create_arc(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        start=arc_angle - 90.0,
                        extent=180.0,
                        style=tk.ARC,
                        outline=color,
                        width=3,
                    )

            if self.show_center_var.get():
                self.canvas.create_line(cx - 5, cy, cx + 5, cy, fill=color, width=2)
                self.canvas.create_line(cx, cy - 5, cx, cy + 5, fill=color, width=2)

            self.canvas.create_text(
                cx + 8,
                cy - 8,
                text=f"p{particle_index + 1}",
                anchor="sw",
                fill=color,
                font=("Segoe UI", 9, "bold"),
            )

    def _format_frame_status(self, frame_index: int) -> str:
        if self.tracks is None or self.diagnostics is None:
            return "Tracking diagnostics will appear here after processing."

        time_s = frame_index / self.fps if self.fps else 0.0
        lines = [f"Frame {frame_index + 1} at {time_s:.4f} s"]
        for particle_index, diag in enumerate(self.diagnostics[frame_index], start=1):
            lines.append(
                f"p{particle_index}: center=({float(diag['center_x']):.2f}, {float(diag['center_y']):.2f}) px, "
                f"score={float(diag['match_score']):.3f}, "
                f"mode={diag.get('mode_name', 'unknown')}"
            )
        return "\n".join(lines)

    def _on_slider_change(self, value: str) -> None:
        if self.tracks is None or self.diagnostics is None:
            return
        self._stop_playback()
        self._render_processed_frame(int(round(float(value))))

    def _step_frame(self, delta: int) -> None:
        if self.tracks is None or self.diagnostics is None:
            return
        self._stop_playback()
        self._render_processed_frame(self.frame_index_var.get() + delta)

    def _toggle_playback(self) -> None:
        if self.play_job is not None:
            self._stop_playback()
            return
        if self.tracks is None or self.diagnostics is None:
            return
        if self.frame_index_var.get() >= len(self.frames) - 1:
            self._render_processed_frame(0)
        self.play_button.configure(text="Pause")
        self._play_next_frame()

    def _play_next_frame(self) -> None:
        if self.tracks is None or self.diagnostics is None:
            self._stop_playback()
            return
        next_frame = self.frame_index_var.get() + 1
        if next_frame >= len(self.frames):
            self._stop_playback()
            return
        self._render_processed_frame(next_frame)
        delay_ms = max(15, int(round(1000.0 / self.fps))) if self.fps else 33
        self.play_job = self.root.after(delay_ms, self._play_next_frame)

    def _stop_playback(self) -> None:
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play")

    def on_canvas_press(self, event: tk.Event[tk.Misc]) -> None:
        if not self.frames or (self.tracks is not None and self.diagnostics is not None):
            return
        self.drag_start = (event.x / self.display_scale, event.y / self.display_scale)
        if self.preview_circle_id is not None:
            self.canvas.delete(self.preview_circle_id)
            self.preview_circle_id = None

    def on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        if (
            not self.frames
            or self.drag_start is None
            or (self.tracks is not None and self.diagnostics is not None)
        ):
            return
        start_x, start_y = self.drag_start
        end_x = event.x / self.display_scale
        end_y = event.y / self.display_scale
        radius = max(2.0, float(np.hypot(end_x - start_x, end_y - start_y)))
        cx = start_x * self.display_scale
        cy = start_y * self.display_scale
        draw_radius = radius * self.display_scale
        angle = float(np.degrees(np.arctan2(-(end_y - start_y), end_x - start_x)))
        if self.preview_circle_id is not None:
            self.canvas.delete(self.preview_circle_id)
        if self.selection_mode_var.get() == "Half circle / arc":
            self.preview_circle_id = self.canvas.create_arc(
                cx - draw_radius,
                cy - draw_radius,
                cx + draw_radius,
                cy + draw_radius,
                start=angle - 90.0,
                extent=180.0,
                style=tk.ARC,
                outline="#66ccff",
                width=2,
                dash=(4, 3),
            )
        else:
            self.preview_circle_id = self.canvas.create_oval(
                cx - draw_radius,
                cy - draw_radius,
                cx + draw_radius,
                cy + draw_radius,
                outline="#66ccff",
                width=2,
                dash=(4, 3),
            )

    def on_canvas_release(self, event: tk.Event[tk.Misc]) -> None:
        if (
            not self.frames
            or self.drag_start is None
            or (self.tracks is not None and self.diagnostics is not None)
        ):
            return
        start_x, start_y = self.drag_start
        end_x = event.x / self.display_scale
        end_y = event.y / self.display_scale
        radius = max(2.0, float(np.hypot(end_x - start_x, end_y - start_y)))
        arc_angle = None
        if self.selection_mode_var.get() == "Half circle / arc":
            arc_angle = float(np.degrees(np.arctan2(-(end_y - start_y), end_x - start_x)))
        self.seed_points.append((float(start_x), float(start_y)))
        self.seed_radii.append(radius)
        self.seed_arc_angles.append(arc_angle)
        self.drag_start = None
        if self.preview_circle_id is not None:
            self.canvas.delete(self.preview_circle_id)
            self.preview_circle_id = None
        self.tracks = None
        self.diagnostics = None
        self.output_table = None
        self.output_headers = None
        self.stiffness_results = []
        self._stop_playback()
        self.frame_index_var.set(0)
        if self.frames:
            self.frame_slider.configure(to=max(0, len(self.frames) - 1), value=0)
            self.frame_label_var.set(f"Frame 1 / {len(self.frames)}")
        self.frame_status_var.set(
            "Selection updated. Run tracking again to inspect diagnostics."
        )
        self._display_frame(self.frames[0])
        self.status_var.set(f"Selected {len(self.seed_points)} particle region(s).")
        self._update_buttons()

    def undo_last(self) -> None:
        if not self.seed_points:
            return
        self.seed_points.pop()
        self.seed_radii.pop()
        self.seed_arc_angles.pop()
        self.tracks = None
        self.diagnostics = None
        self.output_table = None
        self.output_headers = None
        self.stiffness_results = []
        self._stop_playback()
        self.frame_index_var.set(0)
        if self.frames:
            self.frame_slider.configure(to=max(0, len(self.frames) - 1), value=0)
            self.frame_label_var.set(f"Frame 1 / {len(self.frames)}")
        self.frame_status_var.set(
            "Selection updated. Run tracking again to inspect diagnostics."
        )
        self._display_frame(self.frames[0])
        self.status_var.set(f"Selected {len(self.seed_points)} particle(s).")
        self._update_buttons()

    def clear_points(self) -> None:
        self.seed_points = []
        self.seed_radii = []
        self.seed_arc_angles = []
        self.drag_start = None
        self.preview_circle_id = None
        self.tracks = None
        self.diagnostics = None
        self.output_table = None
        self.output_headers = None
        self.stiffness_results = []
        self._stop_playback()
        self.frame_index_var.set(0)
        if self.frames:
            self.frame_slider.configure(to=max(0, len(self.frames) - 1), value=0)
            self.frame_label_var.set(f"Frame 1 / {len(self.frames)}")
        self.frame_status_var.set(
            "Selection cleared. Draw particles first, then run tracking."
        )
        if self.frames:
            self._display_frame(self.frames[0])
        self.status_var.set("Selection cleared.")
        self._update_buttons()

    def _read_positive_float(self, text: str, name: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be a number.") from exc
        if value <= 0:
            raise RuntimeError(f"{name} must be positive.")
        return value

    def _read_positive_int(self, text: str, name: str) -> int:
        try:
            value = int(text)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer.") from exc
        if value <= 0:
            raise RuntimeError(f"{name} must be positive.")
        return value

    def start_tracking(self) -> None:
        if not self.frames or self.video_path is None or self.fps is None:
            messagebox.showerror("No video", "Load a video before tracking.")
            return
        if not self.seed_points:
            messagebox.showerror("No particles", "Draw at least one particle circle first.")
            return

        try:
            pixel_size = self._read_positive_float(
                self.pixel_size_var.get(), "Pixel size"
            )
            temperature_c = self._read_positive_float(
                self.temperature_var.get(), "Temperature"
            )
            roi_radius = self._read_positive_int(
                self.roi_radius_var.get(), "Search radius"
            )
            template_radius = self._read_positive_int(
                self.template_radius_var.get(), "Template radius"
            )
            match_threshold = self._read_positive_float(
                self.match_threshold_var.get(), "Match threshold"
            )
            tracking_mode = {
                "Auto": "auto",
                "Bright spot": "bright",
                "Ring / hollow": "ring",
            }[self.tracking_mode_var.get()]
        except RuntimeError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.status_var.set("Tracking particles...")
        self.root.update_idletasks()
        temperature_k = temperature_c + 273.15

        try:
            tracks, diagnostics = track_selected_particles(
                frames=self.frames,
                seed_points=self.seed_points,
                seed_radii=self.seed_radii,
                seed_arc_angles=self.seed_arc_angles,
                roi_radius=roi_radius,
                template_radius=template_radius,
                blur_size=5,
                refine_radius=max(3, template_radius // 2),
                min_match_score=match_threshold,
                min_contrast=10.0,
                tracking_mode=tracking_mode,
            )
        except Exception as exc:
            self.tracks = None
            self.diagnostics = None
            self.output_table = None
            self.output_headers = None
            self.stiffness_results = []
            self._stop_playback()
            messagebox.showerror("Tracking failed", str(exc))
            self.status_var.set("Tracking failed.")
            self.frame_status_var.set("Tracking stopped before completion.")
            self._update_buttons()
            return

        table, headers = build_output_table(
            tracks=tracks,
            fps=self.fps,
            pixel_size=pixel_size,
        )
        stiffness = estimate_stiffness(
            table=table,
            headers=headers,
            pixel_size_um=pixel_size,
            temperature_k=temperature_k,
        )

        self.tracks = tracks
        self.diagnostics = diagnostics
        self.output_table = table
        self.output_headers = headers
        self.stiffness_results = stiffness
        self.frame_slider.configure(to=max(0, tracks.shape[0] - 1), value=0)
        self.frame_index_var.set(0)
        self._stop_playback()
        self._render_processed_frame(0)
        self._set_results_text(
            self._format_results(
                stiffness,
                tracks,
                pixel_size,
                temperature_c,
                self.tracking_mode_var.get(),
            )
        )
        self.status_var.set(
            f"Tracked {tracks.shape[1]} particle(s) across {tracks.shape[0]} frames."
        )
        self._update_buttons()

    def _format_results(
        self,
        stiffness: list[dict[str, float]],
        tracks: np.ndarray,
        pixel_size: float,
        temperature: float,
        tracking_mode: str,
    ) -> str:
        lines = [
            f"Tracked particles: {tracks.shape[1]}",
            f"Frames: {tracks.shape[0]}",
            f"FPS: {self.fps:.3f}" if self.fps is not None else "FPS: unknown",
            f"Pixel size: {pixel_size:.6f} um/px",
            f"Temperature: {temperature:.2f} C",
            f"Tracking mode: {tracking_mode}",
            "",
            "column\tvariance_px2\tvariance_um2\tk_pN_per_um\tk_pN_per_nm",
        ]
        for row in stiffness:
            lines.append(
                f"{row['column']}\t{row['variance_px2']:.8f}\t"
                f"{row['variance_um2']:.8f}\t{row['k_pn_per_um']:.6f}\t{row['k_pn_per_nm']:.6f}"
            )
        lines.extend(
            [
                "",
                "Use Save .dat to export the tracked coordinates and displacements.",
            ]
        )
        return "\n".join(lines)

    def _set_results_text(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

    def save_dat(self) -> None:
        if (
            self.output_table is None
            or self.output_headers is None
            or self.video_path is None
            or self.fps is None
        ):
            messagebox.showerror("No data", "Track particles before saving.")
            return

        try:
            pixel_size = self._read_positive_float(
                self.pixel_size_var.get(), "Pixel size"
            )
            temperature_c = self._read_positive_float(
                self.temperature_var.get(), "Temperature"
            )
        except RuntimeError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        temperature_k = temperature_c + 273.15

        default_name = self.video_path.stem + "_tracked.dat"
        output = filedialog.asksaveasfilename(
            title="Save displacement data",
            defaultextension=".dat",
            initialfile=default_name,
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
        )
        if not output:
            return

        output_path = Path(output)
        try:
            write_dat(
                output_path=output_path,
                table=self.output_table,
                headers=self.output_headers,
                unit="um",
                pixel_size=pixel_size,
                fps=self.fps,
                video_path=self.video_path,
                temperature_k=temperature_k,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status_var.set(f"Saved results to {output_path.name}.")
        messagebox.showinfo("Saved", f"Saved tracked motion to:\n{output_path}")


def main() -> None:
    root = tk.Tk()
    app = ParticleTrackerApp(root)
    root.minsize(1100, 700)
    root.mainloop()


if __name__ == "__main__":
    main()
