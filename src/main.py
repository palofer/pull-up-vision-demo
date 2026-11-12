"""End-to-end pull-up pose analysis pipeline and CLI entry point."""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import cv2
from ultralytics import YOLO

from angles import (
    Repetition,
    calculate_elbow_angles,
    detect_repetitions,
    enrich_repetition_metrics,
    save_angles_csv,
    save_repetitions_csv,
    save_summary_csv,
    save_summary_json,
)
from config import DEFAULT_CONFIG, AnalysisConfig
from viewer import ViewerInput, save_viewer
from visual import (
    save_dashboard_mp4,
    save_elbow_angles_chart,
    save_shoulder_position_chart,
    save_summary_plots,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT / "data/raw"
PROCESSED_DATA_DIR = ROOT / "data/processed"
DEFAULT_WEIGHTS = "yolo11m-pose.pt"

COCO_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


def get_video_fps(video: Path) -> float:
    """Read video frame rate and raise ValueError if invalid."""
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or fps <= 0:
            raise ValueError(f"Could not read valid FPS from video: {video}")
        return fps
    finally:
        capture.release()


def save_keypoints_csv(pose_frames: list[dict], path: Path) -> None:
    """Export per-frame COCO-17 keypoint coordinates and confidence scores to CSV."""
    columns = [
        "frame",
        "person_idx",
        "keypoint_idx",
        "keypoint",
        "x_px",
        "y_px",
        "keypoint_conf",
        "person_conf",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for frame in pose_frames:
            for person, keypoints in enumerate(frame["xy"]):
                for index, (xy, confidence) in enumerate(
                    zip(keypoints, frame["conf"][person], strict=True)
                ):
                    writer.writerow(
                        {
                            "frame": frame["frame"],
                            "person_idx": person,
                            "keypoint_idx": index,
                            "keypoint": COCO_KEYPOINT_NAMES.get(index, f"keypoint_{index}"),
                            "x_px": float(xy[0]),
                            "y_px": float(xy[1]),
                            "keypoint_conf": float(confidence),
                            "person_conf": float(frame["person_conf"][person]),
                        }
                    )


def resolve_input_video(video_arg: str | Path | None) -> Path:
    """Resolve explicit video path or pick the first video in data/raw/."""
    if video_arg is not None:
        candidate = Path(video_arg).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Input video not found: {candidate}")
        return candidate

    extensions = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
    candidates: list[Path] = []
    if RAW_DATA_DIR.is_dir():
        for pattern in extensions:
            candidates.extend(sorted(RAW_DATA_DIR.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No input video specified and no video files found in {RAW_DATA_DIR}. "
            "Pass a video path: uv run python src/main.py data/raw/<clip>.mp4"
        )
    return candidates[0].resolve()


def run_pipeline(
    video: str | Path | None = None,
    weights: str | Path = DEFAULT_WEIGHTS,
    config: AnalysisConfig = DEFAULT_CONFIG,
    export_mp4: bool = False,
) -> tuple[list[dict], list[dict], list[Repetition], dict[str, Path]]:
    """Run YOLO11-pose inference, repetition grading, and visual artifact generation."""
    video_path = resolve_input_video(video)
    fps = get_video_fps(video_path)
    weights_path = Path(weights).expanduser()
    model_identifier = str(weights_path if weights_path.is_file() else weights)
    model = YOLO(model_identifier)

    output_dir = PROCESSED_DATA_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "keypoints": output_dir / "keypoints.csv",
        "angles": output_dir / "elbow-angles.csv",
        "repetitions": output_dir / "repetitions.csv",
        "summary_csv": output_dir / "summary.csv",
        "summary_json": output_dir / "summary.json",
        "plots": output_dir / "plots.png",
        "elbow_chart": output_dir / "elbow-angles.png",
        "shoulder_chart": output_dir / "shoulder-position.png",
        "viewer": output_dir / "viewer.html",
    }

    pose_frames: list[dict] = []
    for frame_idx, result in enumerate(model(str(video_path), stream=True, verbose=False)):
        pose = result.keypoints
        pose_frames.append(
            {
                "frame": frame_idx,
                "xy": pose.xy.cpu().numpy(),
                "conf": pose.conf.cpu().numpy(),
                "person_conf": result.boxes.conf.cpu().numpy(),
            }
        )

    save_keypoints_csv(pose_frames, paths["keypoints"])
    angles = calculate_elbow_angles(pose_frames, config)
    save_angles_csv(angles, paths["angles"])
    repetitions = detect_repetitions(angles, pose_frames, config)
    repetitions = enrich_repetition_metrics(repetitions, angles, fps, config)
    save_repetitions_csv(repetitions, paths["repetitions"])
    save_summary_csv(repetitions, video_path.name, paths["summary_csv"])
    save_summary_json(
        repetitions,
        video_path.name,
        fps,
        Path(model_identifier).name,
        config,
        paths["summary_json"],
    )

    viewer_input = ViewerInput(
        video=video_path,
        keypoints_csv=paths["keypoints"],
        angles=angles,
        repetitions=repetitions,
        config=config,
    )
    save_elbow_angles_chart(angles, video_path, output_dir, config)
    save_shoulder_position_chart(angles, paths["keypoints"], video_path, output_dir, config)
    save_summary_plots(viewer_input, paths["plots"])
    save_viewer(viewer_input, paths["viewer"])

    if export_mp4:
        mp4_path = output_dir / f"{video_path.stem}-dashboard.mp4"
        save_dashboard_mp4(viewer_input, mp4_path)
        paths["dashboard_mp4"] = mp4_path

    return pose_frames, angles, repetitions, paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pull-up analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze a pull-up video clip with YOLO11-pose and generate visual reports."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to the input video file (defaults to the first video in data/raw/).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Pose weights path or Ultralytics model identifier (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="Render a synchronized split-screen H.264 MP4 video dashboard.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    _, _, detected_reps, output_paths = run_pipeline(
        video=cli_args.video,
        weights=cli_args.weights,
        export_mp4=cli_args.export_mp4,
    )
    print(f"Detected repetitions: {len(detected_reps)}")
    for artifact_path in output_paths.values():
        print(f"Saved: {artifact_path}")
