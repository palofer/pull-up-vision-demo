"""OpenCV/ffmpeg rendering of the split-screen MP4 dashboard."""

import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from angles import RepetitionState, complete_percentage, count_states
from config import DEFAULT_VISUAL_CONFIG
from pose import SKELETON_CONNECTIONS, SKELETON_KEYPOINTS
from viewer import ViewerData, ViewerInput, prepare_viewer_data

matplotlib.use("Agg")

PointsMap = dict[int, tuple[int, int, float]]


class _StateStyle(NamedTuple):
    label: str
    bgr_color: tuple[int, int, int]
    hex_color: str


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB hex string to an OpenCV BGR tuple."""
    return int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)


def _state_styles(data: ViewerData) -> dict[RepetitionState, _StateStyle]:
    colors = data["chart"]["colors"]["states"]
    labels = data["chart"]["labels"]["states"]
    return {
        state: _StateStyle(label.upper(), _hex_to_bgr(colors[state]), colors[state])
        for state, label in labels.items()
    }


def _draw_skeleton(
    image: np.ndarray,
    points: PointsMap,
    min_confidence: float,
    *,
    scale: float,
    offset_x: int,
    connection_colors: dict[tuple[int, int], tuple[int, int, int]],
    line_thickness: int = 3,
    point_radius: int = 6,
) -> None:
    """Render visible skeleton joints and bones above the confidence threshold."""

    def transform(x: int, y: int) -> tuple[int, int]:
        return round((x - offset_x) * scale), round(y * scale)

    for a, b in SKELETON_CONNECTIONS:
        if (
            a not in points
            or b not in points
            or points[a][2] < min_confidence
            or points[b][2] < min_confidence
        ):
            continue
        start = transform(*points[a][:2])
        end = transform(*points[b][:2])
        color = connection_colors.get((a, b), (255, 255, 255))
        cv2.line(image, start, end, color, line_thickness, lineType=cv2.LINE_AA)

    for idx in SKELETON_KEYPOINTS:
        if idx not in points:
            continue
        orig_x, orig_y, conf = points[idx]
        x, y = transform(orig_x, orig_y)
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        color = (255, 255, 255) if conf >= min_confidence else (68, 68, 239)
        cv2.circle(image, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (x, y), point_radius, (17, 24, 39), 1, lineType=cv2.LINE_AA)


def _mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


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

    total_height = DEFAULT_VISUAL_CONFIG.dashboard_height_px
    scale = total_height / crop["height"]
    video_width = round(crop["width"] * scale)
    video_width += video_width % 2
    panel_width = DEFAULT_VISUAL_CONFIG.dashboard_panel_width_px
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
            frame["earClearancePx"] if frame["earClearancePx"] is not None else np.nan
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
    span = spec["yMax"] - spec["yMin"]
    label_y = spec["yMax"] - span * DEFAULT_VISUAL_CONFIG.label_offset_ratio

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
    ax_ears.axhline(0, color=colors["grip"], linewidth=1.0, label=labels["grip"])
    ax_ears.axhline(
        data["clearanceThresholdPx"],
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
    ax_elbow.set(
        xlabel="Time (s)",
        ylabel="Elbow (°)",
        ylim=(viewer_input.config.rom_extension_reference_deg, 0),
    )
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
            viewer_input.config.keypoint_confidence,
            scale=scale,
            offset_x=crop["x"],
            connection_colors=skeleton_colors,
            line_thickness=4,
            point_radius=5,
        )

        panel = np.full((total_height, panel_width, 3), (255, 255, 255), dtype=np.uint8)
        completed_reps = [r for r in reps if frame_idx >= r["endFrame"]]
        counts = count_states(completed_reps)
        full_count = counts["complete"]
        pct = complete_percentage(completed_reps)
        returned_reps = [
            r for r in completed_reps if r["state"] in ("complete", "partial")
        ]
        avg_ascent = _mean([r["ascent"] for r in returned_reps])
        avg_descent = _mean([r["descent"] for r in returned_reps])
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
        pct_str = f"{pct:.0f}%" if pct is not None else "--"
        cv2.putText(
            panel,
            f"Complete: {full_count}/{counts['complete'] + counts['partial']} ({pct_str})",
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

        ascent_str = f"{avg_ascent:.2f}s" if avg_ascent is not None else "--"
        descent_str = f"{avg_descent:.2f}s" if avg_descent is not None else "--"
        no_ext_str = f"  No ext: {counts['no_extension']}"
        cv2.putText(
            panel,
            f"Avg ascent: {ascent_str}   Avg descent: {descent_str}{no_ext_str}",
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
        y_chart = DEFAULT_VISUAL_CONFIG.dashboard_chart_top_px
        x_chart = max(10, (panel_width - ch_w) // 2)
        panel[y_chart : y_chart + ch_h, x_chart : x_chart + ch_w] = chart_bgr

        y_table = y_chart + ch_h + DEFAULT_VISUAL_CONFIG.table_header_gap_px
        headers = ["REP", "STATE", "BOTTOM", "TOP", "ASCENT", "DESCENT"]
        xs = DEFAULT_VISUAL_CONFIG.table_column_xs
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
        for i, rep in enumerate(completed_reps[: DEFAULT_VISUAL_CONFIG.table_max_rows]):
            y = y_table + DEFAULT_VISUAL_CONFIG.table_row_height_px * (i + 1)
            if y > total_height - 15:
                break
            style = state_styles[rep["state"]]
            row_cells = [
                str(rep["number"]),
                style.label,
                f"{rep['bottom']:.0f} deg" if rep["bottom"] is not None else "--",
                f"{rep['top']:.0f} deg" if rep["top"] is not None else "--",
                f"{rep['ascent']:.2f}s" if rep["ascent"] is not None else "--",
                f"{rep['descent']:.2f}s" if rep["descent"] is not None else "--",
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
