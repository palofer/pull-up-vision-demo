import numpy as np
import pytest

from src.angles import calculate_angle, calculate_ear_height_above_grip


@pytest.fixture
def mock_pose_data():
    """Build minimal 17-keypoint COCO pose payloads for unit testing."""

    def _builder(
        keypoints_by_idx: dict[int, tuple[list[float], float]], frame_id: int = 0
    ) -> list[dict]:
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.zeros((1, 17), dtype=np.float32)
        for idx, (coords, confidence) in keypoints_by_idx.items():
            xy[0, idx] = coords
            conf[0, idx] = confidence
        return [
            {
                "frame": frame_id,
                "xy": xy,
                "conf": conf,
                "person_conf": np.array([0.95], dtype=np.float32),
            }
        ]

    return _builder


def test_calculate_angle_right_angle():
    shoulder = np.array([0.0, 1.0])
    elbow = np.array([0.0, 0.0])
    wrist = np.array([1.0, 0.0])

    assert calculate_angle(shoulder, elbow, wrist) == pytest.approx(90.0)


def test_calculate_angle_straight_arm_and_acute_flexion():
    shoulder = np.array([0.0, 1.0])
    elbow = np.array([0.0, 0.0])
    wrist_extended = np.array([0.0, -1.0])
    wrist_acute = np.array([1.0, 1.0])

    assert calculate_angle(shoulder, elbow, wrist_extended) == pytest.approx(180.0)
    assert calculate_angle(shoulder, elbow, wrist_acute) == pytest.approx(45.0)


def test_calculate_angle_degenerate_zero_length_vector_returns_none():
    point = np.array([10.0, 20.0])
    other = np.array([15.0, 25.0])

    assert calculate_angle(point, point, other) is None
    assert calculate_angle(other, point, point) is None


def test_ear_height_above_grip_nominal(mock_pose_data):
    points = {
        3: ([320.0, 80.0], 0.98),
        4: ([300.0, 80.0], 0.97),
        9: ([290.0, 130.0], 0.93),
        10: ([350.0, 130.0], 0.94),
    }
    data = mock_pose_data(points, frame_id=0)

    assert calculate_ear_height_above_grip(data) == {0: pytest.approx(50.0)}


def test_ear_height_above_grip_single_ear_above_confidence_threshold(mock_pose_data):
    points = {
        3: ([320.0, 80.0], 0.20),
        4: ([300.0, 80.0], 0.97),
        9: ([290.0, 130.0], 0.93),
        10: ([350.0, 130.0], 0.94),
    }
    data = mock_pose_data(points, frame_id=0)

    assert calculate_ear_height_above_grip(data) == {0: pytest.approx(50.0)}


def test_ear_height_above_grip_missing_ears_returns_empty(mock_pose_data):
    points = {
        3: ([320.0, 80.0], 0.15),
        4: ([300.0, 80.0], 0.10),
        9: ([290.0, 130.0], 0.93),
        10: ([350.0, 130.0], 0.94),
    }
    data = mock_pose_data(points, frame_id=0)

    assert calculate_ear_height_above_grip(data) == {}


def test_detect_and_enrich_repetitions_complete_partial_and_no_extension(mock_pose_data):
    from src.angles import (
        build_summary_metrics,
        detect_repetitions,
        enrich_repetition_metrics,
    )

    # Synthetic sequence at 10 FPS:
    # Rep 1 (frames 0..4): extended -> flexed (ears +30px above grip) -> extended => complete
    # Rep 2 (frames 4..8): extended -> flexed (ears +5px above grip) -> extended => partial
    # Rep 3 (frames 8..10): extended -> flexed, ends without extension => no_extension
    angle_trajectories = [
        (0, 160.0, 160.0, 150.0),  # hang (ears at y=150, grip at y=100 => -50px)
        (1, 120.0, 122.0, 110.0),  # pull start
        (2, 55.0, 55.0, 70.0),     # peak 1 (ears at y=70 => +30px above grip >= 20px)
        (3, 110.0, 115.0, 110.0),  # lowering
        (4, 160.0, 160.0, 150.0),  # return to extension (end of rep 1, onset of rep 2)
        (5, 115.0, 118.0, 120.0),  # pull 2 start
        (6, 80.0, 85.0, 95.0),     # peak 2 (ears at y=95 => +5px above grip < 20px)
        (7, 125.0, 125.0, 120.0),  # lowering
        (8, 155.0, 155.0, 150.0),  # return to extension (end of rep 2, onset of rep 3)
        (9, 110.0, 110.0, 100.0),  # pull 3 start
        (10, 70.0, 70.0, 75.0),    # clip ends mid-rep without returning to >=150°
    ]

    angles = []
    pose_frames = []
    for frame_id, left_deg, right_deg, ear_y in angle_trajectories:
        angles.append(
            {
                "frame": frame_id,
                "person_idx": 0,
                "left_elbow_deg": left_deg,
                "left_min_conf": 0.95,
                "right_elbow_deg": right_deg,
                "right_min_conf": 0.95,
            }
        )
        pose_frames.extend(
            mock_pose_data(
                {
                    3: ([300.0, ear_y], 0.95),
                    4: ([320.0, ear_y], 0.95),
                    9: ([280.0, 100.0], 0.95),
                    10: ([340.0, 100.0], 0.95),
                },
                frame_id=frame_id,
            )
        )

    reps = detect_repetitions(angles, pose_frames)
    enriched = enrich_repetition_metrics(reps, angles, fps=10.0)

    assert [r["state"] for r in enriched] == ["complete", "partial", "no_extension"]

    # Rep 1: onset=0, peak=2, end=4 -> ascent=0.2s, descent=0.2s, sweep=160-55=105° -> 105/125=84.0%
    assert enriched[0]["onset_frame"] == 0
    assert enriched[0]["peak_frame"] == 2
    assert enriched[0]["end_frame"] == 4
    assert enriched[0]["max_clearance_px"] == pytest.approx(30.0)
    assert enriched[0]["ascent_time_s"] == pytest.approx(0.2)
    assert enriched[0]["descent_time_s"] == pytest.approx(0.2)
    assert enriched[0]["rom_pct"] == pytest.approx(84.0)

    # Rep 2: partial (max_clearance_px = 5.0 < 20.0)
    assert enriched[1]["max_clearance_px"] == pytest.approx(5.0)
    assert enriched[1]["ascent_time_s"] == pytest.approx(0.2)
    assert enriched[1]["descent_time_s"] == pytest.approx(0.2)

    # Rep 3: no_extension has no descent time
    assert enriched[2]["descent_time_s"] is None

    summary = build_summary_metrics(enriched, "sample.mp4")
    assert summary["total_repetitions"] == 3
    assert summary["complete_repetitions"] == 1
    assert summary["partial_repetitions"] == 1
    assert summary["no_extension_repetitions"] == 1
    assert summary["avg_ascent_time_s"] == pytest.approx(0.2)
    assert summary["avg_descent_time_s"] == pytest.approx(0.2)


def test_count_states_and_complete_percentage_exclude_no_extension():
    from src.angles import complete_percentage, count_states

    reps = [
        {"state": "complete"},
        {"state": "partial"},
        {"state": "no_extension"},
        {"state": "complete"},
    ]
    assert count_states(reps) == {"complete": 2, "partial": 1, "no_extension": 1}
    assert complete_percentage(reps) == pytest.approx(2 / 3 * 100.0)

    only_no_extension = [{"state": "no_extension"}]
    assert complete_percentage(only_no_extension) is None
    assert complete_percentage([]) is None


def test_prepare_viewer_data_maps_frames_and_repetitions(tmp_path):
    import csv

    import cv2

    from src.angles import detect_repetitions, enrich_repetition_metrics
    from src.viewer import ViewerInput, prepare_viewer_data

    video_path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    for _ in range(5):
        frame = np.full((120, 160, 3), 120, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    keypoints_csv = tmp_path / "keypoints.csv"
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
    ear_ys = [90.0, 75.0, 50.0, 75.0, 90.0]
    with keypoints_csv.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=columns)
        csv_writer.writeheader()
        for frame_idx, ear_y in enumerate(ear_ys):
            for kp_idx, (x_val, y_val) in [
                (3, (75.0, ear_y)),
                (4, (85.0, ear_y)),
                (5, (65.0, ear_y + 15.0)),
                (6, (95.0, ear_y + 15.0)),
                (9, (60.0, 80.0)),
                (10, (100.0, 80.0)),
            ]:
                csv_writer.writerow(
                    {
                        "frame": frame_idx,
                        "person_idx": 0,
                        "keypoint_idx": kp_idx,
                        "keypoint": f"kp_{kp_idx}",
                        "x_px": x_val,
                        "y_px": y_val,
                        "keypoint_conf": 0.95,
                        "person_conf": 0.95,
                    }
                )

    angles = [
        {"frame": 0, "person_idx": 0, "left_elbow_deg": 160.0, "right_elbow_deg": 160.0},
        {"frame": 1, "person_idx": 0, "left_elbow_deg": 120.0, "right_elbow_deg": 120.0},
        {"frame": 2, "person_idx": 0, "left_elbow_deg": 55.0, "right_elbow_deg": 55.0},
        {"frame": 3, "person_idx": 0, "left_elbow_deg": 115.0, "right_elbow_deg": 115.0},
        {"frame": 4, "person_idx": 0, "left_elbow_deg": 160.0, "right_elbow_deg": 160.0},
    ]
    pose_frames = []
    for frame_idx, ear_y in enumerate(ear_ys):
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.zeros((1, 17), dtype=np.float32)
        for kp_idx, coords in [
            (3, [75.0, ear_y]),
            (4, [85.0, ear_y]),
            (9, [60.0, 80.0]),
            (10, [100.0, 80.0]),
        ]:
            xy[0, kp_idx] = coords
            conf[0, kp_idx] = 0.95
        pose_frames.append(
            {
                "frame": frame_idx,
                "xy": xy,
                "conf": conf,
                "person_conf": np.array([0.95], dtype=np.float32),
            }
        )

    reps = enrich_repetition_metrics(
        detect_repetitions(angles, pose_frames), angles, fps=10.0
    )
    payload = prepare_viewer_data(
        ViewerInput(
            video=video_path,
            keypoints_csv=keypoints_csv,
            angles=angles,
            repetitions=reps,
        )
    )

    assert payload["fps"] == pytest.approx(10.0)
    assert len(payload["frames"]) == 5
    assert payload["frames"][2]["earClearancePx"] == pytest.approx(30.0)
    assert len(payload["repetitions"]) == 1
    assert payload["repetitions"][0]["state"] == "complete"
    assert payload["repetitions"][0]["top"] == pytest.approx(55.0)
    assert payload["repetitions"][0]["bottom"] == pytest.approx(160.0)


def test_visual_exports_static_charts(tmp_path):
    import csv

    import cv2

    from src.angles import detect_repetitions, enrich_repetition_metrics
    from src.viewer import ViewerInput
    from src.visual import (
        save_elbow_angles_chart,
        save_shoulder_position_chart,
        save_summary_plots,
    )

    video_path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    for _ in range(5):
        writer.write(np.full((120, 160, 3), 120, dtype=np.uint8))
    writer.release()

    keypoints_csv = tmp_path / "keypoints.csv"
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
    ear_ys = [90.0, 75.0, 50.0, 75.0, 90.0]
    with keypoints_csv.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=columns)
        csv_writer.writeheader()
        for frame_idx, ear_y in enumerate(ear_ys):
            for kp_idx, (x_val, y_val) in [
                (3, (75.0, ear_y)),
                (4, (85.0, ear_y)),
                (5, (65.0, ear_y + 15.0)),
                (6, (95.0, ear_y + 15.0)),
                (9, (60.0, 80.0)),
                (10, (100.0, 80.0)),
            ]:
                csv_writer.writerow(
                    {
                        "frame": frame_idx,
                        "person_idx": 0,
                        "keypoint_idx": kp_idx,
                        "keypoint": f"kp_{kp_idx}",
                        "x_px": x_val,
                        "y_px": y_val,
                        "keypoint_conf": 0.95,
                        "person_conf": 0.95,
                    }
                )

    angles = [
        {"frame": 0, "person_idx": 0, "left_elbow_deg": 160.0, "right_elbow_deg": 160.0},
        {"frame": 1, "person_idx": 0, "left_elbow_deg": 120.0, "right_elbow_deg": 120.0},
        {"frame": 2, "person_idx": 0, "left_elbow_deg": 55.0, "right_elbow_deg": 55.0},
        {"frame": 3, "person_idx": 0, "left_elbow_deg": 115.0, "right_elbow_deg": 115.0},
        {"frame": 4, "person_idx": 0, "left_elbow_deg": 160.0, "right_elbow_deg": 160.0},
    ]
    pose_frames = []
    for frame_idx, ear_y in enumerate(ear_ys):
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.zeros((1, 17), dtype=np.float32)
        for kp_idx, coords in [
            (3, [75.0, ear_y]),
            (4, [85.0, ear_y]),
            (9, [60.0, 80.0]),
            (10, [100.0, 80.0]),
        ]:
            xy[0, kp_idx] = coords
            conf[0, kp_idx] = 0.95
        pose_frames.append(
            {
                "frame": frame_idx,
                "xy": xy,
                "conf": conf,
                "person_conf": np.array([0.95], dtype=np.float32),
            }
        )

    reps = enrich_repetition_metrics(
        detect_repetitions(angles, pose_frames), angles, fps=10.0
    )
    viewer_input = ViewerInput(
        video=video_path,
        keypoints_csv=keypoints_csv,
        angles=angles,
        repetitions=reps,
    )

    elbow_png = save_elbow_angles_chart(angles, video_path, tmp_path)
    shoulder_png = save_shoulder_position_chart(angles, keypoints_csv, video_path, tmp_path)
    plots_png = save_summary_plots(viewer_input, tmp_path / "plots.png")

    assert elbow_png.is_file() and elbow_png.stat().st_size > 0
    assert shoulder_png.is_file() and shoulder_png.stat().st_size > 0
    assert plots_png.is_file() and plots_png.stat().st_size > 0


def test_main_cli_parsing_and_keypoints_export(tmp_path):
    from src.main import DEFAULT_WEIGHTS, parse_args, save_keypoints_csv

    assert DEFAULT_WEIGHTS == "yolo11m-pose.pt"

    args = parse_args(["data/raw/sample.mp4", "--export-mp4"])
    assert args.video == "data/raw/sample.mp4"
    assert args.weights == "yolo11m-pose.pt"
    assert args.export_mp4 is True

    kp_path = tmp_path / "keypoints.csv"
    xy = np.zeros((1, 17, 2), dtype=np.float32)
    xy[0, 5] = [100.0, 200.0]
    conf = np.full((1, 17), 0.9, dtype=np.float32)
    save_keypoints_csv(
        [{"frame": 0, "xy": xy, "conf": conf, "person_conf": np.array([0.95])}],
        kp_path,
    )
    content = kp_path.read_text(encoding="utf-8")
    assert "left_shoulder" in content
    assert "100.0,200.0" in content




