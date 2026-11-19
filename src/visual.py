"""Static diagnostic plots (matplotlib PNGs)."""

import csv
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from angles import AngleRow
from config import DEFAULT_CONFIG, DEFAULT_VISUAL_CONFIG, AnalysisConfig
from viewer import ViewerInput, prepare_viewer_data

matplotlib.use("Agg")


def save_elbow_angles_chart(
    angles: list[AngleRow],
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
    angles: list[AngleRow],
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
        [f["earClearancePx"] if f["earClearancePx"] is not None else np.nan for f in frames]
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
        span = data["chart"]["yMax"] - data["chart"]["yMin"]
        y_label = data["chart"]["yMax"] - span * DEFAULT_VISUAL_CONFIG.label_offset_ratio
        ax_ears.text(
            (t0 + t1) / 2,
            y_label,
            f"#{rep['number']}",
            color=c,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_ears.axhline(0, color=colors["grip"], linewidth=1.2, label="Grip reference (0 px)")
    ax_ears.axhline(
        data["clearanceThresholdPx"],
        color=colors["threshold"],
        linewidth=1.3,
        label=f"Clearance threshold (+{data['clearanceThresholdPx']:g} px)",
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
    ax_elbows.set_ylim(viewer_input.config.rom_extension_reference_deg, 0)
    ax_elbows.set_title("Elbow Angles (180° dead hang at bottom, 0° peak flexion at top)")
    ax_elbows.grid(True, alpha=0.3)
    ax_elbows.legend(loc="upper right", fontsize=8.5)

    ax_shoulders.axhline(0, color=colors["grip"], linewidth=1.2, label="Grip (0 px)")
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
