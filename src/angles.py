"""Geometric calculations, repetition state machine, and tabular/JSON exports."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from config import DEFAULT_CONFIG, AnalysisConfig

RepetitionState = Literal["complete", "partial", "no_extension"]


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
    rom_pct: float | str
    left_rom_pct: float | str
    right_rom_pct: float | str
    ascent_time_s: float | str
    descent_time_s: float | str


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
    pose_frames: list[dict], config: AnalysisConfig = DEFAULT_CONFIG
) -> list[dict]:
    """Compute bilateral elbow angles per frame for the highest-confidence person."""
    angles: list[dict] = []
    sides = [("left", 5, 7, 9), ("right", 6, 8, 10)]
    for frame in pose_frames:
        row: dict[str, int | float | str] = {"frame": frame["frame"], "person_idx": ""}
        if frame["xy"].shape[0]:
            person = int(np.argmax(frame["person_conf"]))
            row["person_idx"] = person
            xy, conf = frame["xy"][person], frame["conf"][person]
            for side, shoulder, elbow, wrist in sides:
                confidence = float(np.min(conf[[shoulder, elbow, wrist]]))
                angle = calculate_angle(xy[shoulder], xy[elbow], xy[wrist])
                row[f"{side}_elbow_deg"] = (
                    angle if confidence >= config.keypoint_confidence and angle is not None else ""
                )
                row[f"{side}_min_conf"] = confidence
        angles.append(row)
    return angles


def calculate_ear_height_above_grip(
    pose_frames: list[dict], config: AnalysisConfig = DEFAULT_CONFIG
) -> dict[int, float]:
    """Vertical distance from ears to median wrist reference (positive = ears above grip)."""
    ear_heights: dict[int, float] = {}
    wrist_heights: list[float] = []
    for frame in pose_frames:
        if frame["xy"].shape[0] == 0:
            continue
        person = int(np.argmax(frame["person_conf"]))
        xy, conf = frame["xy"][person], frame["conf"][person]
        ears = [xy[i] for i in (3, 4) if conf[i] >= config.clearance_confidence]
        wrists = [xy[i] for i in (9, 10) if conf[i] >= config.clearance_confidence]
        if ears:
            ear_heights[int(frame["frame"])] = float(np.mean([p[1] for p in ears]))
        if wrists:
            wrist_heights.append(float(np.mean([p[1] for p in wrists])))
    if not wrist_heights:
        return {}
    grip_height = float(np.median(wrist_heights))
    return {frame: grip_height - height for frame, height in ear_heights.items()}


def detect_repetitions(
    angles: list[dict],
    pose_frames: list[dict],
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
    angles: list[dict],
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
            else ""
        )
        ascent_time = (int(rep["peak_frame"]) - onset) / fps if onset is not None else ""
        descent_time = (end - int(rep["peak_frame"])) / fps
        updated: Repetition = dict(rep)
        updated.update(
            {
                "rom_pct": rom_pct,
                "left_rom_pct": rom_by_side.get("left", ""),
                "right_rom_pct": rom_by_side.get("right", ""),
                "ascent_time_s": ascent_time,
                "descent_time_s": "" if rep["state"] == "no_extension" else descent_time,
            }
        )
        enriched.append(updated)
    return enriched


def save_angles_csv(angles: list[dict], path: Path) -> None:
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
                        rep.get("onset_frame") if rep.get("onset_frame") is not None else ""
                    ),
                    "peak_frame": rep["peak_frame"],
                    "end_frame": rep["end_frame"],
                    "min_left_angle_deg": round(rep["min_left_angle_deg"], 1),
                    "min_right_angle_deg": round(rep["min_right_angle_deg"], 1),
                    "max_clearance_px": round(clearance, 1) if clearance is not None else "",
                    "state": rep["state"],
                    "rom_pct": (
                        round(float(rep["rom_pct"]), 1) if rep.get("rom_pct", "") != "" else ""
                    ),
                    "left_rom_pct": (
                        round(float(rep["left_rom_pct"]), 1)
                        if rep.get("left_rom_pct", "") != ""
                        else ""
                    ),
                    "right_rom_pct": (
                        round(float(rep["right_rom_pct"]), 1)
                        if rep.get("right_rom_pct", "") != ""
                        else ""
                    ),
                    "ascent_time_s": (
                        round(float(rep["ascent_time_s"]), 2)
                        if rep.get("ascent_time_s", "") != ""
                        else ""
                    ),
                    "descent_time_s": (
                        round(float(rep["descent_time_s"]), 2)
                        if rep.get("descent_time_s", "") != ""
                        else ""
                    ),
                }
            )


def build_summary_metrics(repetitions: list[Repetition], video_name: str) -> dict[str, object]:
    """Compute aggregate counts and average ROM/tempo across completed repetitions."""
    states: tuple[RepetitionState, ...] = ("complete", "partial", "no_extension")
    by_state = {state: sum(rep["state"] == state for rep in repetitions) for state in states}
    returned_reps = [rep for rep in repetitions if rep["state"] in ("complete", "partial")]

    def _mean(field: str, decimals: int) -> float | None:
        values = [float(rep[field]) for rep in returned_reps if rep.get(field, "") != ""]
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
                "rom_pct": (
                    round(float(rep["rom_pct"]), 1) if rep.get("rom_pct", "") != "" else None
                ),
                "left_rom_pct": (
                    round(float(rep["left_rom_pct"]), 1)
                    if rep.get("left_rom_pct", "") != ""
                    else None
                ),
                "right_rom_pct": (
                    round(float(rep["right_rom_pct"]), 1)
                    if rep.get("right_rom_pct", "") != ""
                    else None
                ),
                "ascent_time_s": (
                    round(float(rep["ascent_time_s"]), 2)
                    if rep.get("ascent_time_s", "") != ""
                    else None
                ),
                "descent_time_s": (
                    round(float(rep["descent_time_s"]), 2)
                    if rep.get("descent_time_s", "") != ""
                    else None
                ),
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
