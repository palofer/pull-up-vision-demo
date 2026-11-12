"""Static diagnostic plots and optional H.264 split-screen MP4 dashboard export."""

import csv
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from angles import RepetitionState
from config import DEFAULT_CONFIG, AnalysisConfig
from viewer import ViewerData, ViewerInput, prepare_viewer_data

matplotlib.use("Agg")

PointsMap = dict[int, tuple[int, int, float]]
Connection = tuple[int, int]

_SKELETON_CONNECTIONS: tuple[Connection, ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class _StateStyle(NamedTuple):
    label: str
    bgr_color: tuple[int, int, int]
    hex_color: str


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB hex string to an OpenCV BGR tuple."""
    return int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)


def _draw_skeleton(
    image: np.ndarray,
    points: PointsMap,
    connections: tuple[Connection, ...] | list[Connection],
    min_confidence: float,
    *,
    default_connection_color: tuple[int, int, int],
    point_color: tuple[int, int, int],
    low_confidence_color: tuple[int, int, int],
    scale: float = 1.0,
    offset_x: int = 0,
    connection_colors: dict[Connection, tuple[int, int, int]] | None = None,
    line_thickness: int = 3,
    point_radius: int = 6,
    point_border: bool = False,
) -> None:
    """Render visible pose keypoints and bones above the confidence threshold."""

    def transform(x: int, y: int) -> tuple[int, int]:
        return round((x - offset_x) * scale), round(y * scale)

    for a, b in connections:
        if (
            a not in points
            or b not in points
            or points[a][2] < min_confidence
            or points[b][2] < min_confidence
        ):
            continue
        start = transform(*points[a][:2])
        end = transform(*points[b][:2])
        color = (connection_colors or {}).get((a, b), default_connection_color)
        cv2.line(image, start, end, color, line_thickness, lineType=cv2.LINE_AA)

    for idx in range(5, 17):
        if idx not in points:
            continue
        orig_x, orig_y, conf = points[idx]
        x, y = transform(orig_x, orig_y)
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        color = point_color if conf >= min_confidence else low_confidence_color
        cv2.circle(image, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)
        if point_border:
            cv2.circle(image, (x, y), point_radius, (17, 24, 39), 1, lineType=cv2.LINE_AA)


def _state_styles(data: ViewerData) -> dict[RepetitionState, _StateStyle]:
    colors = data["chart"]["colors"]["states"]
    labels = data["chart"]["labels"]["states"]
    return {
        state: _StateStyle(label.upper(), _hex_to_bgr(colors[state]), colors[state])
        for state, label in labels.items()
    }


def _count_states(
    states: Iterable[RepetitionState], styles: dict[RepetitionState, _StateStyle]
) -> dict[RepetitionState, int]:
    counts = dict.fromkeys(styles, 0)
    for state in states:
        counts[state] += 1
    return counts


def save_elbow_angles_chart(
    angles: list[dict],
    video: Path,
    output_dir: Path,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> Path:
    """Export a static PNG chart of bilateral elbow angles over time."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError("Could not read valid video FPS")
    time_s = np.array([int(row["frame"]) for row in angles]) / fps
    series = {}
    for side, label, color in [
        ("left", "Left elbow", "#2563eb"),
        ("right", "Right elbow", "#f59e0b"),
    ]:
        series[label] = (
            np.array(
                [
                    float(row[f"{side}_elbow_deg"])
                    if row.get(f"{side}_elbow_deg", "") != ""
                    else np.nan
                    for row in angles
                ]
            ),
            color,
        )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for label, (values, color) in series.items():
        ax.plot(time_s, values, label=label, color=color, linewidth=1.8)
    ax.axhline(
        config.extension_threshold_deg,
        color="#6b7280",
        linestyle="--",
        linewidth=1.2,
        label=f"Extension threshold ({config.extension_threshold_deg:g}°)",
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Elbow angle (°)",
        ylim=(config.rom_extension_reference_deg, 0),
    )
    ax.set_title("Bilateral Elbow Angle Trajectory (180° extended at bottom, 0° flexed at top)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "elbow-angles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_shoulder_position_chart(
    angles: list[dict],
    keypoints_csv: Path,
    video: Path,
    output_dir: Path,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> Path:
    """Plot vertical shoulder positions and frame-by-frame vertical difference |Δy|."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError("Could not read valid video FPS")

    person_by_frame = {
        int(f["frame"]): int(f["person_idx"])
        for f in angles
        if f.get("person_idx", "") != ""
    }
    total_frames = len(angles)
    left_shoulder = np.full(total_frames, np.nan)
    right_shoulder = np.full(total_frames, np.nan)
    with keypoints_csv.open(newline="", encoding="utf-8") as file:
        for point in csv.DictReader(file):
            frame = int(point["frame"])
            idx = int(point["keypoint_idx"])
            if idx not in (5, 6) or frame >= total_frames:
                continue
            if int(point["person_idx"]) != person_by_frame.get(frame, -1):
                continue
            if float(point["keypoint_conf"]) < config.keypoint_confidence:
                continue
            y = float(point["y_px"])
            (left_shoulder if idx == 5 else right_shoulder)[frame] = y

    time_s = np.arange(total_frames) / fps
    diff_px = np.abs(left_shoulder - right_shoulder)
    fig, (ax_pos, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    ax_pos.plot(time_s, left_shoulder, label="Left shoulder", color="#2563eb", linewidth=1.8)
    ax_pos.plot(time_s, right_shoulder, label="Right shoulder", color="#f59e0b", linewidth=1.8)
    ax_pos.set_ylabel("Vertical image coordinate y (px)")
    ax_pos.set_title("Vertical Shoulder Trajectory")
    ax_pos.invert_yaxis()
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="upper right")

    ax_diff.plot(time_s, diff_px, color="#7e22ce", label="|Δy| between shoulders", linewidth=1.5)
    ax_diff.fill_between(time_s, diff_px, alpha=0.15, color="#7e22ce")
    ax_diff.set(xlabel="Time (s)", ylabel="Difference (px)")
    ax_diff.set_title("Observed Vertical Shoulder Tilt |Δy| (exploratory, no alert threshold)")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend(loc="upper right")
    fig.tight_layout()
    path = output_dir / "shoulder-position.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_summary_plots(viewer_input: ViewerInput, output_path: Path) -> Path:
    """Export a high-resolution 3-panel summary chart matching the interactive viewer."""
    data = prepare_viewer_data(viewer_input)
    fps = data["fps"]
    frames = data["frames"]
    time_s = np.arange(len(frames)) / fps
    ears = np.array(
        [f["earsAboveBarPx"] if f["earsAboveBarPx"] is not None else np.nan for f in frames]
    )
    left_elbow = np.array(
        [f["leftElbow"] if f["leftElbow"] is not None else np.nan for f in frames]
    )
    right_elbow = np.array(
        [f["rightElbow"] if f["rightElbow"] is not None else np.nan for f in frames]
    )
    left_sh = np.array(
        [f["leftShoulderPx"] if f["leftShoulderPx"] is not None else np.nan for f in frames]
    )
    right_sh = np.array(
        [f["rightShoulderPx"] if f["rightShoulderPx"] is not None else np.nan for f in frames]
    )

    colors = data["chart"]["colors"]
    state_colors = colors["states"]

    fig, (ax_ears, ax_elbows, ax_shoulders) = plt.subplots(
        3,
        1,
        figsize=(12, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1.3, 1.0, 1.0)},
    )

    for rep in data["repetitions"]:
        t0, t1 = rep["startFrame"] / fps, rep["endFrame"] / fps
        c = state_colors[rep["state"]]
        for ax in (ax_ears, ax_elbows, ax_shoulders):
            ax.axvspan(t0, t1, color=c, alpha=0.12)
        y_label = data["chart"]["yMax"] - (data["chart"]["yMax"] - data["chart"]["yMin"]) * 0.08
        ax_ears.text(
            (t0 + t1) / 2,
            y_label,
            f"#{rep['number']}",
            color=c,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_ears.axhline(0, color=colors["bar"], linewidth=1.2, label="Grip reference (0 px)")
    ax_ears.axhline(
        data["thresholdAboveBar"],
        color=colors["threshold"],
        linewidth=1.3,
        label=f"Clearance threshold (+{data['thresholdAboveBar']:g} px)",
    )
    ax_ears.plot(time_s, ears, color=colors["series"], linewidth=2.0, label="Ears above grip")
    ax_ears.set_ylabel("Distance from grip (px)")
    ax_ears.set_ylim(data["chart"]["yMin"], data["chart"]["yMax"])
    ax_ears.set_title("Ear Height Relative to Grip Reference")
    ax_ears.grid(True, alpha=0.3)
    ax_ears.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    ax_elbows.axhline(
        data["extensionAngle"],
        color=colors["extension"],
        linestyle="--",
        linewidth=1.2,
        label=f"Extension ({data['extensionAngle']:g}°)",
    )
    ax_elbows.plot(time_s, left_elbow, color="#2563eb", linewidth=1.7, label="Left elbow")
    ax_elbows.plot(time_s, right_elbow, color="#f59e0b", linewidth=1.7, label="Right elbow")
    ax_elbows.set_ylabel("Elbow angle (°)")
    ax_elbows.set_ylim(180, 0)
    ax_elbows.set_title("Elbow Angles (180° dead hang at bottom, 0° peak flexion at top)")
    ax_elbows.grid(True, alpha=0.3)
    ax_elbows.legend(loc="upper right", fontsize=8.5)

    ax_shoulders.axhline(0, color=colors["bar"], linewidth=1.2, label="Grip (0 px)")
    ax_shoulders.plot(time_s, left_sh, color="#2563eb", linewidth=1.7, label="Left shoulder")
    ax_shoulders.plot(time_s, right_sh, color="#f59e0b", linewidth=1.7, label="Right shoulder")
    ax_shoulders.set_ylabel("Height vs. grip (px)")
    ax_shoulders.set_xlabel("Time (s)")
    ax_shoulders.set_ylim(data["shoulderRange"]["min"], data["shoulderRange"]["max"])
    ax_shoulders.set_title("Shoulder Position Relative to Grip")
    ax_shoulders.grid(True, alpha=0.3)
    ax_shoulders.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_dashboard_mp4(viewer_input: ViewerInput, output_path: Path) -> Path:
    """Render a browser-compatible H.264 split-screen MP4 video synchronized with the charts."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required on PATH to export an H.264 MP4 dashboard")
    video = Path(viewer_input.video).expanduser().resolve()
    data = prepare_viewer_data(viewer_input)
    state_styles = _state_styles(data)
    angles_by_frame = {int(f["frame"]): f for f in viewer_input.angles}
    points_by_frame = {
        frame["frame"]: {
            idx: (round(pt[0]), round(pt[1]), pt[2])
            for idx, pt in enumerate(frame["keypoints"])
            if pt is not None
        }
        for frame in data["frames"]
    }

    cap = cv2.VideoCapture(str(video))
    fps = data["fps"]
    crop = data["crop"]
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("Could not open source video or read FPS")

    total_height = 1024
    scale = total_height / crop["height"]
    video_width = round(crop["width"] * scale)
    video_width += video_width % 2
    panel_width = 680
    total_width = video_width + panel_width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.stem}-silent.mp4")
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_width, total_height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize OpenCV VideoWriter")

    reps = data["repetitions"]
    time_s = np.arange(len(data["frames"])) / fps
    ears = np.array(
        [
            frame["earsAboveBarPx"] if frame["earsAboveBarPx"] is not None else np.nan
            for frame in data["frames"]
        ]
    )
    left_elbow = np.array(
        [
            frame["leftElbow"] if frame["leftElbow"] is not None else np.nan
            for frame in data["frames"]
        ]
    )
    right_elbow = np.array(
        [
            frame["rightElbow"] if frame["rightElbow"] is not None else np.nan
            for frame in data["frames"]
        ]
    )

    spec = data["chart"]
    colors = spec["colors"]
    labels = spec["labels"]
    label_y = spec["yMax"] - (spec["yMax"] - spec["yMin"]) * 0.08

    fig, (ax_ears, ax_elbow) = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.4),
        dpi=100,
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )
    fig.patch.set_facecolor("#ffffff")
    for ax in (ax_ears, ax_elbow):
        ax.set_facecolor("#ffffff")
        ax.tick_params(colors=colors["tickText"], labelsize=8)
        ax.xaxis.label.set_color(colors["axisText"])
        ax.yaxis.label.set_color(colors["axisText"])
        ax.grid(True, alpha=0.4, color=colors["grid"])

    (ears_line,) = ax_ears.plot([], [], color=colors["series"], linewidth=2.0, label="Ears")
    ax_ears.axhline(0, color=colors["bar"], linewidth=1.0, label=labels["bar"])
    ax_ears.axhline(
        data["thresholdAboveBar"],
        color=colors["threshold"],
        linewidth=1.2,
        label=labels["threshold"],
    )
    ax_ears.set(
        ylabel="From grip (px)",
        ylim=(spec["yMin"], spec["yMax"]),
        xlim=(0, max(time_s[-1], 0.1)),
    )
    ax_ears.set_title(labels["title"], color=colors["threshold"], fontsize=10.5, fontweight="bold")
    ax_ears.legend(loc="upper right", fontsize=7, frameon=False)
    vline_ears = ax_ears.axvline(x=0, color=colors["cursor"], linestyle="--", linewidth=1.2)
    (ears_dot,) = ax_ears.plot([], [], "o", color=colors["series"], markersize=4.5)

    (left_line,) = ax_elbow.plot([], [], color="#2563eb", linewidth=1.6, label="Left")
    (right_line,) = ax_elbow.plot([], [], color="#f59e0b", linewidth=1.6, label="Right")
    ax_elbow.axhline(
        data["extensionAngle"],
        color=colors["extension"],
        linestyle="--",
        linewidth=1.1,
        label="Extension",
    )
    ax_elbow.set(xlabel="Time (s)", ylabel="Elbow (°)", ylim=(180, 0))
    ax_elbow.set_title("Elbow Angle", color=colors["threshold"], fontsize=10.5, fontweight="bold")
    ax_elbow.legend(loc="upper right", fontsize=7, frameon=False)
    vline_elbow = ax_elbow.axvline(x=0, color=colors["cursor"], linestyle="--", linewidth=1.2)

    fig.tight_layout()
    next_zone_idx = 0

    skeleton_colors = {
        (5, 7): (238, 211, 34),
        (7, 9): (238, 211, 34),
        (6, 8): (11, 158, 245),
        (8, 10): (11, 158, 245),
        (5, 6): (255, 255, 255),
        (5, 11): (255, 255, 255),
        (6, 12): (255, 255, 255),
        (11, 12): (255, 255, 255),
        (11, 13): (182, 114, 244),
        (13, 15): (182, 114, 244),
        (12, 14): (68, 68, 239),
        (14, 16): (68, 68, 239),
    }

    frame_idx = 0
    while True:
        ok, frame_img = cap.read()
        if not ok:
            break
        cropped = frame_img[
            crop["y"] : crop["y"] + crop["height"],
            crop["x"] : crop["x"] + crop["width"],
        ]
        resized = cv2.resize(cropped, (video_width, total_height))

        points = points_by_frame.get(frame_idx, {})
        _draw_skeleton(
            resized,
            points,
            _SKELETON_CONNECTIONS,
            viewer_input.config.keypoint_confidence,
            default_connection_color=(255, 255, 255),
            point_color=(255, 255, 255),
            low_confidence_color=(68, 68, 239),
            scale=scale,
            offset_x=crop["x"],
            connection_colors=skeleton_colors,
            line_thickness=4,
            point_radius=5,
            point_border=True,
        )

        panel = np.full((total_height, panel_width, 3), (255, 255, 255), dtype=np.uint8)
        completed_reps = [r for r in reps if frame_idx >= r["endFrame"]]
        counts = _count_states((r["state"] for r in completed_reps), state_styles)
        full_count = counts["complete"]
        pct = round(full_count / len(completed_reps) * 100) if completed_reps else 0
        rom_color = (
            state_styles["complete"].bgr_color
            if full_count
            else state_styles["partial"].bgr_color
        )
        ink_bgr = (39, 24, 17)
        muted_bgr = (128, 114, 107)

        cv2.putText(
            panel,
            f"Reps: {len(completed_reps)}",
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            ink_bgr,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Full ROM: {full_count}/{len(completed_reps)} ({pct}%)",
            (200, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            rom_color,
            2,
            lineType=cv2.LINE_AA,
        )

        angle_row = angles_by_frame.get(frame_idx, {})
        l_val, r_val = angle_row.get("left_elbow_deg", ""), angle_row.get("right_elbow_deg", "")
        valid_angles = [float(v) for v in (l_val, r_val) if v != ""]
        mean_elbow = sum(valid_angles) / len(valid_angles) if valid_angles else None
        elbow_txt = f"Elbows: {mean_elbow:.0f} deg" if mean_elbow is not None else "Elbows: --"
        cv2.putText(
            panel,
            elbow_txt,
            (panel_width - 210, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ink_bgr,
            1,
            lineType=cv2.LINE_AA,
        )

        avg_pull = data["averages"]["pull"]
        avg_lower = data["averages"]["lower"]
        pull_str = f"{avg_pull:.2f}s" if avg_pull is not None else "--"
        lower_str = f"{avg_lower:.2f}s" if avg_lower is not None else "--"
        cv2.putText(
            panel,
            f"Avg ascent: {pull_str}   Avg descent: {lower_str}",
            (30, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            muted_bgr,
            1,
            lineType=cv2.LINE_AA,
        )

        current_t = frame_idx / fps
        up_to = min(frame_idx + 1, len(time_s))
        ears_line.set_data(time_s[:up_to], ears[:up_to])
        left_line.set_data(time_s[:up_to], left_elbow[:up_to])
        right_line.set_data(time_s[:up_to], right_elbow[:up_to])

        while next_zone_idx < len(reps) and frame_idx >= reps[next_zone_idx]["endFrame"]:
            rep = reps[next_zone_idx]
            c = state_styles[rep["state"]].hex_color
            t0, t1 = rep["startFrame"] / fps, rep["endFrame"] / fps
            ax_ears.axvspan(t0, t1, color=c, alpha=0.2)
            ax_elbow.axvspan(t0, t1, color=c, alpha=0.2)
            ax_ears.text(
                (t0 + t1) / 2,
                label_y,
                str(rep["number"]),
                color=c,
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
            next_zone_idx += 1

        cur_ear = ears[frame_idx] if frame_idx < len(ears) else np.nan
        if np.isfinite(cur_ear):
            ears_dot.set_data([current_t], [cur_ear])
        else:
            ears_dot.set_data([], [])
        vline_ears.set_xdata([current_t, current_t])
        vline_elbow.set_xdata([current_t, current_t])

        fig.canvas.draw()
        chart_rgba = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        chart_bgr = cv2.cvtColor(chart_rgba, cv2.COLOR_RGB2BGR)
        ch_h, ch_w = chart_bgr.shape[:2]
        y_chart = 92
        x_chart = max(10, (panel_width - ch_w) // 2)
        panel[y_chart : y_chart + ch_h, x_chart : x_chart + ch_w] = chart_bgr

        y_table = y_chart + ch_h + 28
        headers = ["REP", "STATE", "BOTTOM", "TOP", "ASCENT", "DESCENT"]
        xs = [30, 95, 245, 355, 455, 565]
        for x, col in zip(xs, headers, strict=False):
            cv2.putText(
                panel,
                col,
                (x, y_table),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                muted_bgr,
                1,
                lineType=cv2.LINE_AA,
            )
        for i, rep in enumerate(completed_reps[:10]):
            y = y_table + 30 * (i + 1)
            if y > total_height - 15:
                break
            style = state_styles[rep["state"]]
            row_cells = [
                str(rep["number"]),
                style.label,
                f"{rep['bottom']:.0f} deg" if rep["bottom"] is not None else "--",
                f"{rep['top']:.0f} deg" if rep["top"] is not None else "--",
                f"{rep['pull']:.2f}s" if rep["pull"] is not None else "--",
                f"{rep['lower']:.2f}s" if rep["lower"] is not None else "--",
            ]
            cell_colors = [ink_bgr, style.bgr_color, ink_bgr, ink_bgr, ink_bgr, ink_bgr]
            for x, txt, color in zip(xs, row_cells, cell_colors, strict=False):
                cv2.putText(
                    panel,
                    txt,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    1,
                    lineType=cv2.LINE_AA,
                )

        canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        canvas[:, :video_width] = resized
        canvas[:, video_width:] = panel
        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()
    plt.close(fig)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temp_path),
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    temp_path.unlink(missing_ok=True)
    return output_path
