"""Self-contained interactive HTML viewer generation for a pull-up set."""

import csv
import html
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import cv2
import numpy as np

from angles import AngleRow, Repetition, RepetitionState
from config import DEFAULT_CONFIG, DEFAULT_VISUAL_CONFIG, AnalysisConfig
from pose import EAR_INDICES, KEYPOINT_COUNT, SHOULDER_INDICES, WRIST_INDICES


@dataclass(frozen=True)
class ViewerInput:
    """Input artifacts required to build the HTML viewer or MP4 dashboard."""

    video: Path
    keypoints_csv: Path
    angles: list[AngleRow]
    repetitions: list[Repetition]
    config: AnalysisConfig = DEFAULT_CONFIG


class _ChartColors(TypedDict):
    series: str
    grip: str
    threshold: str
    extension: str
    cursor: str
    grid: str
    tickText: str
    axisText: str
    states: dict[RepetitionState, str]


class _ChartLabels(TypedDict):
    title: str
    xAxis: str
    yAxis: str
    series: str
    grip: str
    threshold: str
    states: dict[RepetitionState, str]


class _ViewerChart(TypedDict):
    yMin: float
    yMax: float
    colors: _ChartColors
    labels: _ChartLabels


class _ViewerFrame(TypedDict):
    frame: int
    earClearancePx: float | None
    leftShoulderPx: float | None
    rightShoulderPx: float | None
    leftElbow: float | None
    rightElbow: float | None
    keypoints: list[list[float] | None]


class _ViewerRepetition(TypedDict):
    number: int
    startFrame: int
    endFrame: int
    state: RepetitionState
    bottom: float | None
    top: float | None
    ascent: float | None
    descent: float | None


class _ViewerRange(TypedDict):
    min: float
    max: float


class _ViewerCrop(TypedDict):
    x: int
    y: int
    width: int
    height: int


class ViewerData(TypedDict):
    fps: float
    width: int
    height: int
    crop: _ViewerCrop
    gripHeight: float | None
    clearanceThresholdPx: float
    extensionAngle: float
    confidence: float
    chart: _ViewerChart
    frames: list[_ViewerFrame]
    repetitions: list[_ViewerRepetition]
    shoulderRange: _ViewerRange


CHART_PALETTE: _ChartColors = {
    "series": "#2563eb",
    "grip": "#d1d5db",
    "threshold": "#111827",
    "extension": "#6b7280",
    "cursor": "#ef4444",
    "grid": "#e5e7eb",
    "tickText": "#6b7280",
    "axisText": "#4b5563",
    "states": {
        "complete": "#15803d",
        "partial": "#b45309",
        "no_extension": "#7e22ce",
    },
}
STATE_LABELS: dict[RepetitionState, str] = {
    "complete": "Complete",
    "partial": "Partial",
    "no_extension": "No extension",
}


def _chart_specification(frames: list[_ViewerFrame], threshold: float) -> _ViewerChart:
    values = [
        float(frame["earClearancePx"])
        for frame in frames
        if frame["earClearancePx"] is not None
    ]
    references = [0.0, threshold, *values]
    minimum, maximum = min(references), max(references)
    margin = max(
        DEFAULT_VISUAL_CONFIG.chart_margin_px,
        (maximum - minimum) * DEFAULT_VISUAL_CONFIG.chart_margin_ratio,
    )
    return {
        "yMin": minimum - margin,
        "yMax": maximum + margin,
        "colors": CHART_PALETTE,
        "labels": {
            "title": "Ear height relative to grip",
            "xAxis": "Time (s)",
            "yAxis": "Distance from grip (px)",
            "series": "Ears",
            "grip": "Grip (0 px)",
            "threshold": f"Above grip (+{threshold:g} px)",
            "states": STATE_LABELS,
        },
    }


def _shoulder_range(frames: list[_ViewerFrame]) -> _ViewerRange:
    values = [
        value
        for frame in frames
        for value in (frame["leftShoulderPx"], frame["rightShoulderPx"])
        if value is not None
    ]
    minimum, maximum = min([0.0, *values]), max([0.0, *values])
    margin = max(
        DEFAULT_VISUAL_CONFIG.chart_margin_px,
        (maximum - minimum) * DEFAULT_VISUAL_CONFIG.chart_margin_ratio,
    )
    return {"min": minimum - margin, "max": maximum + margin}


def _mean_height(
    points: dict[int, tuple[float, float, float]], indices: tuple[int, ...], confidence: float
) -> float | None:
    heights = [points[i][1] for i in indices if i in points and points[i][2] >= confidence]
    return sum(heights) / len(heights) if heights else None


def _height_above_grip(
    points: dict[int, tuple[float, float, float]],
    index: int,
    grip_height: float | None,
    confidence: float,
) -> float | None:
    if grip_height is None or index not in points or points[index][2] < confidence:
        return None
    return grip_height - points[index][1]


def _float_or_none(value: object) -> float | None:
    return float(value) if value not in (None, "") else None


def _repetition_statistics(
    repetitions: list[Repetition], angles_by_frame: dict[int, AngleRow]
) -> list[_ViewerRepetition]:
    stats: list[_ViewerRepetition] = []
    for number, rep in enumerate(repetitions, start=1):
        start, end = int(rep["start_frame"]), int(rep["end_frame"])
        rows = [angles_by_frame.get(frame, {}) for frame in range(start, end + 1)]
        values = {
            side: [
                float(row[field])
                for row in rows
                if (field := f"{side}_elbow_deg") in row and row[field] != ""
            ]
            for side in ("left", "right")
        }
        stats.append(
            {
                "number": number,
                "startFrame": start,
                "endFrame": end,
                "state": rep["state"],
                "bottom": max((max(v) for v in values.values() if v), default=None),
                "top": min((min(v) for v in values.values() if v), default=None),
                "ascent": _float_or_none(rep.get("ascent_time_s")),
                "descent": _float_or_none(rep.get("descent_time_s")),
            }
        )
    return stats


def _content_crop(
    video: Path, width: int, height: int, total_frames: int, config: AnalysisConfig
) -> _ViewerCrop:
    """Detect active horizontal content columns and discard lateral black bars."""
    capture = cv2.VideoCapture(str(video))
    column_means: list[np.ndarray] = []
    for index in np.linspace(0, max(total_frames - 1, 0), config.crop_sample_count).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, image = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        column_means.append(gray.mean(axis=0))
    capture.release()
    full: _ViewerCrop = {"x": 0, "y": 0, "width": width, "height": height}
    if not column_means:
        return full
    columns = np.flatnonzero(np.mean(column_means, axis=0) > config.black_bar_threshold)
    if columns.size == 0:
        return full
    return {
        "x": int(columns[0]),
        "y": 0,
        "width": int(columns[-1] - columns[0] + 1),
        "height": height,
    }


def prepare_viewer_data(viewer_input: ViewerInput) -> ViewerData:
    """Assemble synchronized frame, chart, and repetition data for the viewer."""
    video = Path(viewer_input.video).expanduser().resolve()
    config = viewer_input.config
    angles = viewer_input.angles
    repetitions = viewer_input.repetitions
    angles_by_frame = {int(row["frame"]): row for row in angles}
    person_by_frame = {
        frame: int(row["person_idx"])
        for frame, row in angles_by_frame.items()
        if row.get("person_idx", "") != ""
    }
    points_by_frame: dict[int, dict[int, tuple[float, float, float]]] = {}
    with Path(viewer_input.keypoints_csv).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame, person = int(row["frame"]), int(row["person_idx"])
            if person != person_by_frame.get(frame, -1):
                continue
            points_by_frame.setdefault(frame, {})[int(row["keypoint_idx"])] = (
                float(row["x_px"]),
                float(row["y_px"]),
                float(row["keypoint_conf"]),
            )

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Could not read valid video metadata")
    total_frames = max(total_frames, max(angles_by_frame, default=-1) + 1)
    crop = _content_crop(video, width, height, total_frames, config)

    wrist_heights = [
        h
        for points in points_by_frame.values()
        if (h := _mean_height(points, WRIST_INDICES, config.clearance_confidence)) is not None
    ]
    bar_height = statistics.median(wrist_heights) if wrist_heights else None
    frames: list[_ViewerFrame] = []
    for frame in range(total_frames):
        points = points_by_frame.get(frame, {})
        ear_height = _mean_height(points, EAR_INDICES, config.clearance_confidence)
        angle = angles_by_frame.get(frame, {})
        left = _float_or_none(angle.get("left_elbow_deg"))
        right = _float_or_none(angle.get("right_elbow_deg"))
        ears_above_bar = (
            bar_height - ear_height
            if bar_height is not None and ear_height is not None
            else None
        )
        keypoints = [[*points[i]] if i in points else None for i in range(KEYPOINT_COUNT)]
        frames.append(
            {
                "frame": frame,
                "earClearancePx": ears_above_bar,
                "leftShoulderPx": _height_above_grip(
                    points, SHOULDER_INDICES[0], bar_height, config.keypoint_confidence
                ),
                "rightShoulderPx": _height_above_grip(
                    points, SHOULDER_INDICES[1], bar_height, config.keypoint_confidence
                ),
                "leftElbow": left,
                "rightElbow": right,
                "keypoints": keypoints,
            }
        )

    viewer_reps = _repetition_statistics(repetitions, angles_by_frame)
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "crop": crop,
        "gripHeight": bar_height,
        "clearanceThresholdPx": config.clearance_threshold_px,
        "extensionAngle": config.extension_threshold_deg,
        "confidence": config.keypoint_confidence,
        "chart": _chart_specification(frames, config.clearance_threshold_px),
        "frames": frames,
        "repetitions": viewer_reps,
        "shoulderRange": _shoulder_range(frames),
    }


def save_viewer(
    viewer_input: ViewerInput,
    output_path: Path,
) -> Path:
    """Write a self-contained HTML viewer with synchronized video, skeleton, charts, and table."""
    video = Path(viewer_input.video).expanduser().resolve()
    output_path = Path(output_path)
    data = prepare_viewer_data(viewer_input)
    try:
        relative_video = Path(os.path.relpath(video, output_path.resolve().parent))
    except ValueError:
        relative_video = video
    video_url = quote(relative_video.as_posix(), safe="/.:_-")
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("viewer-template.html").read_text(encoding="utf-8")
    document = template.replace("__VIDEO_SRC__", html.escape(video_url, quote=True))
    document = document.replace("__DATA__", data_json)
    colors = data["chart"]["colors"]
    color_replacements = {
        "__COLOR_THRESHOLD__": colors["threshold"],
        "__COLOR_EXTENSION__": colors["extension"],
        "__COLOR_SERIES__": colors["series"],
        "__COLOR_GRIP__": colors["grip"],
        "__COLOR_AXIS_TEXT__": colors["axisText"],
        "__COLOR_COMPLETE__": colors["states"]["complete"],
        "__COLOR_PARTIAL__": colors["states"]["partial"],
        "__COLOR_NO_EXTENSION__": colors["states"]["no_extension"],
    }
    for placeholder, color in color_replacements.items():
        document = document.replace(placeholder, color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path
