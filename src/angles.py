"""Geometric calculations, repetition state machine, and tabular/JSON exports."""

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np

from config import DEFAULT_CONFIG, AnalysisConfig
from pose import EAR_INDICES, SIDE_JOINTS, WRIST_INDICES, PoseFrame

RepetitionState = Literal["complete", "partial", "no_extension"]


class AngleRow(TypedDict):
    """Per-frame bilateral elbow angles and the minimum confidence per side."""

    frame: int
    person_idx: int | str
    left_elbow_deg: float | str
    left_min_conf: float
    right_elbow_deg: float | str
    right_min_conf: float


class HasState(Protocol):
    """Structural type for any record carrying a repetition state (``Repetition``, viewer reps)."""

    state: RepetitionState


# Cycles that returned to extension: the only states eligible for ROM/tempo aggregates.
RETURNED_STATES: tuple[RepetitionState, ...] = ("complete", "partial")

ALL_STATES: tuple[RepetitionState, ...] = ("complete", "partial", "no_extension")


class Repetition(TypedDict, total=False):
    start_frame: int
    onset_frame: int | None
    peak_frame: int
    end_frame: int
    min_left_angle_deg: float
    min_right_angle_deg: float
    best_max_angle_deg: float
    max_clearance_px: float | None
    state: RepetitionState
    rom_pct: float | None
    left_rom_pct: float | None
    right_rom_pct: float | None
    ascent_time_s: float | None
    descent_time_s: float | None


def count_states(repetitions: Sequence[HasState]) -> dict[RepetitionState, int]:
    """Count cycles per state, always returning every state key."""
    counts: dict[RepetitionState, int] = {state: 0 for state in ALL_STATES}
    for rep in repetitions:
        counts[rep["state"]] += 1
    return counts


def complete_percentage(repetitions: Sequence[HasState]) -> float | None:
    """Complete cycles as a percentage of returned (complete + partial) cycles.

    ``no_extension`` cycles are excluded from the denominator, matching the
    glossary: they are their own category, not partials. Returns ``None`` when
    no returned cycle exists.
    """
    counts = count_states(repetitions)
    returned = counts["complete"] + counts["partial"]
    return counts["complete"] / returned * 100.0 if returned else None


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Return angle ABC in degrees, where 180° represents a straight line."""
    vector_a = a - b
    vector_c = c - b
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_c) == 0:
        return None
    cross_product = vector_a[0] * vector_c[1] - vector_a[1] * vector_c[0]
    dot_product = np.dot(vector_a, vector_c)
    return float(np.degrees(np.arctan2(abs(cross_product), dot_product)))


def calculate_elbow_angles(
    pose_frames: list[PoseFrame], config: AnalysisConfig = DEFAULT_CONFIG
) -> list[AngleRow]:
    """Compute bilateral elbow angles per frame for the highest-confidence person."""
    angles: list[AngleRow] = []
    for frame in pose_frames:
        row: dict[str, int | float | str] = {"frame": frame["frame"], "person_idx": ""}
        if frame["xy"].shape[0]:
            person = int(np.argmax(frame["person_conf"]))
            row["person_idx"] = person
            xy, conf = frame["xy"][person], frame["conf"][person]
            for side, (shoulder, elbow, wrist) in SIDE_JOINTS.items():
                confidence = float(np.min(conf[[shoulder, elbow, wrist]]))
                angle = calculate_angle(xy[shoulder], xy[elbow], xy[wrist])
                row[f"{side}_elbow_deg"] = (
                    angle if confidence >= config.keypoint_confidence and angle is not None else ""
                )
                row[f"{side}_min_conf"] = confidence
        angles.append(row)
    return angles


def calculate_ear_height_above_grip(
    pose_frames: list[PoseFrame], config: AnalysisConfig = DEFAULT_CONFIG
) -> dict[int, float]:
    """Vertical distance from ears to median wrist reference (positive = ears above grip)."""
    ear_heights: dict[int, float] = {}
    wrist_heights: list[float] = []
    for frame in pose_frames:
        if frame["xy"].shape[0] == 0:
            continue
        person = int(np.argmax(frame["person_conf"]))
        xy, conf = frame["xy"][person], frame["conf"][person]
        ears = [xy[i] for i in EAR_INDICES if conf[i] >= config.clearance_confidence]
        wrists = [xy[i] for i in WRIST_INDICES if conf[i] >= config.clearance_confidence]
        if ears:
            ear_heights[int(frame["frame"])] = float(np.mean([p[1] for p in ears]))
        if wrists:
            wrist_heights.append(float(np.mean([p[1] for p in wrists])))
    if not wrist_heights:
        return {}
    grip_height = float(np.median(wrist_heights))
    return {frame: grip_height - height for frame, height in ear_heights.items()}


def detect_repetitions(
    angles: list[AngleRow],
    pose_frames: list[PoseFrame],
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> list[Repetition]:
    """Detect pull-up cycles from elbow flexion/extension and grade ear clearance above grip."""
    ear_clearance_by_frame = calculate_ear_height_above_grip(pose_frames, config)
    repetitions: list[Repetition] = []
    current: Repetition | None = None
    last_valid_frame: int | None = None
    last_extended_frame: int | None = None

    for row in angles:
        frame = int(row["frame"])
        if row.get("left_elbow_deg", "") == "" or row.get("right_elbow_deg", "") == "":
            continue
        left = float(row["left_elbow_deg"])
        right = float(row["right_elbow_deg"])
        is_extended = (
            left >= config.extension_threshold_deg and right >= config.extension_threshold_deg
        )
        is_initiated = min(left, right) <= config.initiation_threshold_deg

        if current is None and is_extended:
            last_extended_frame = frame
        if current is None and not is_extended and is_initiated:
            current = {
                "start_frame": frame,
                "onset_frame": last_extended_frame,
                "min_left_angle_deg": left,
                "min_right_angle_deg": right,
                "best_max_angle_deg": max(left, right),
                "peak_frame": frame,
            }
        if current is None:
            continue

        current["min_left_angle_deg"] = min(current["min_left_angle_deg"], left)
        current["min_right_angle_deg"] = min(current["min_right_angle_deg"], right)
        if max(left, right) < current["best_max_angle_deg"]:
            current["best_max_angle_deg"] = max(left, right)
            current["peak_frame"] = frame
        last_valid_frame = frame

        if is_extended:
            current["end_frame"] = frame
            cycle_clearances = [
                ear_clearance_by_frame[f]
                for f in range(current["start_frame"], frame + 1)
                if f in ear_clearance_by_frame
            ]
            current["max_clearance_px"] = max(cycle_clearances) if cycle_clearances else None
            current["state"] = (
                "complete"
                if current["max_clearance_px"] is not None
                and current["max_clearance_px"] >= config.clearance_threshold_px
                else "partial"
            )
            repetitions.append(current)
            current = None
            last_extended_frame = frame

    if current is not None and last_valid_frame is not None:
        current["end_frame"] = last_valid_frame
        cycle_clearances = [
            ear_clearance_by_frame[f]
            for f in range(current["start_frame"], last_valid_frame + 1)
            if f in ear_clearance_by_frame
        ]
        current["max_clearance_px"] = max(cycle_clearances) if cycle_clearances else None
        current["state"] = "no_extension"
        repetitions.append(current)

    return repetitions


def enrich_repetition_metrics(
    repetitions: list[Repetition],
    angles: list[AngleRow],
    fps: float,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> list[Repetition]:
    """Attach bilateral ROM percentage and ascent/descent durations to each detected cycle."""
    if fps <= 0:
        raise ValueError("FPS must be strictly greater than zero")

    reference_sweep = config.rom_extension_reference_deg - config.rom_flexion_reference_deg
    enriched: list[Repetition] = []
    for rep in repetitions:
        onset = rep.get("onset_frame")
        end = int(rep["end_frame"])
        cycle_rows = (
            [row for row in angles if onset <= int(row["frame"]) <= end]
            if onset is not None
            else []
        )
        rom_by_side: dict[str, float] = {}
        for side in ("left", "right"):
            values = [
                float(row[f"{side}_elbow_deg"])
                for row in cycle_rows
                if row.get(f"{side}_elbow_deg", "") != ""
            ]
            if values:
                rom_by_side[side] = (max(values) - min(values)) / reference_sweep * 100.0

        rom_pct = (
            (rom_by_side["left"] + rom_by_side["right"]) / 2.0
            if len(rom_by_side) == 2
            else None
        )
        ascent_time = (int(rep["peak_frame"]) - onset) / fps if onset is not None else None
        descent_time = (end - int(rep["peak_frame"])) / fps
        updated: Repetition = dict(rep)
        updated.update(
            {
                "rom_pct": rom_pct,
                "left_rom_pct": rom_by_side.get("left"),
                "right_rom_pct": rom_by_side.get("right"),
                "ascent_time_s": ascent_time,
                "descent_time_s": None if rep["state"] == "no_extension" else descent_time,
            }
        )
        enriched.append(updated)
    return enriched


def _round_or_empty(value: float | None, ndigits: int) -> float | str:
    """Round a metric for CSV output, mapping missing values to an empty cell."""
    return round(value, ndigits) if value is not None else ""


def _round_or_none(value: float | None, ndigits: int) -> float | None:
    """Round a metric for JSON output, mapping missing values to ``null``."""
    return round(value, ndigits) if value is not None else None


def save_angles_csv(angles: list[AngleRow], path: Path) -> None:
    """Export per-frame bilateral elbow angles and keypoint confidence scores to CSV."""
    columns = [
        "frame",
        "person_idx",
        "left_elbow_deg",
        "left_min_conf",
        "right_elbow_deg",
        "right_min_conf",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(angles)


def save_repetitions_csv(repetitions: list[Repetition], path: Path) -> None:
    """Export detected repetition cycles, grades, ROM percentages, and phase timings to CSV."""
    columns = [
        "start_frame",
        "onset_frame",
        "peak_frame",
        "end_frame",
        "min_left_angle_deg",
        "min_right_angle_deg",
        "max_clearance_px",
        "state",
        "rom_pct",
        "left_rom_pct",
        "right_rom_pct",
        "ascent_time_s",
        "descent_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for rep in repetitions:
            clearance = rep.get("max_clearance_px")
            writer.writerow(
                {
                    "start_frame": rep["start_frame"],
                    "onset_frame": (
                        rep["onset_frame"] if rep.get("onset_frame") is not None else ""
                    ),
                    "peak_frame": rep["peak_frame"],
                    "end_frame": rep["end_frame"],
                    "min_left_angle_deg": round(rep["min_left_angle_deg"], 1),
                    "min_right_angle_deg": round(rep["min_right_angle_deg"], 1),
                    "max_clearance_px": round(clearance, 1) if clearance is not None else "",
                    "state": rep["state"],
                    "rom_pct": _round_or_empty(rep.get("rom_pct"), 1),
                    "left_rom_pct": _round_or_empty(rep.get("left_rom_pct"), 1),
                    "right_rom_pct": _round_or_empty(rep.get("right_rom_pct"), 1),
                    "ascent_time_s": _round_or_empty(rep.get("ascent_time_s"), 2),
                    "descent_time_s": _round_or_empty(rep.get("descent_time_s"), 2),
                }
            )


def build_summary_metrics(repetitions: list[Repetition], video_name: str) -> dict[str, object]:
    """Aggregate counts and average ROM/tempo across returned (complete + partial) cycles."""
    by_state = count_states(repetitions)
    returned_reps = [rep for rep in repetitions if rep["state"] in RETURNED_STATES]

    def _mean(field: str, decimals: int) -> float | None:
        values = [float(rep[field]) for rep in returned_reps if rep.get(field) is not None]
        return round(sum(values) / len(values), decimals) if values else None

    return {
        "video": video_name,
        "total_repetitions": len(repetitions),
        "complete_repetitions": by_state["complete"],
        "partial_repetitions": by_state["partial"],
        "no_extension_repetitions": by_state["no_extension"],
        "avg_rom_pct": _mean("rom_pct", 1),
        "avg_ascent_time_s": _mean("ascent_time_s", 2),
        "avg_descent_time_s": _mean("descent_time_s", 2),
    }


def save_summary_csv(repetitions: list[Repetition], video_name: str, path: Path) -> None:
    """Export a single-row aggregate CSV summary for a video."""
    summary = build_summary_metrics(repetitions, video_name)
    columns = [
        "video",
        "total_repetitions",
        "complete_repetitions",
        "partial_repetitions",
        "no_extension_repetitions",
        "avg_rom_pct",
        "avg_ascent_time_s",
        "avg_descent_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in summary.items()})


def save_summary_json(
    repetitions: list[Repetition],
    video_name: str,
    fps: float,
    model_name: str,
    config: AnalysisConfig,
    path: Path,
) -> None:
    """Export structured JSON report including run metadata, thresholds, and per-rep metrics."""
    summary = build_summary_metrics(repetitions, video_name)
    serialized_reps = []
    for idx, rep in enumerate(repetitions, start=1):
        clearance = rep.get("max_clearance_px")
        serialized_reps.append(
            {
                "rep_number": idx,
                "state": rep["state"],
                "start_frame": rep["start_frame"],
                "onset_frame": rep.get("onset_frame"),
                "peak_frame": rep["peak_frame"],
                "end_frame": rep["end_frame"],
                "min_left_angle_deg": round(rep["min_left_angle_deg"], 1),
                "min_right_angle_deg": round(rep["min_right_angle_deg"], 1),
                "max_clearance_px": round(clearance, 1) if clearance is not None else None,
                "rom_pct": _round_or_none(rep.get("rom_pct"), 1),
                "left_rom_pct": _round_or_none(rep.get("left_rom_pct"), 1),
                "right_rom_pct": _round_or_none(rep.get("right_rom_pct"), 1),
                "ascent_time_s": _round_or_none(rep.get("ascent_time_s"), 2),
                "descent_time_s": _round_or_none(rep.get("descent_time_s"), 2),
            }
        )
    payload = {
        "video": video_name,
        "fps": round(fps, 3),
        "model": model_name,
        "config": asdict(config),
        "summary": summary,
        "repetitions": serialized_reps,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
