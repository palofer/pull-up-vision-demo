"""COCO-17 keypoint topology and naming shared across the pipeline.

This is the single source of truth for keypoint indices, so analysis and
rendering code never hard-codes a joint number or a skeleton edge.
"""

from typing import Final, TypedDict

import numpy as np

# Keypoint indices (COCO-17).
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

KEYPOINT_COUNT = 17

EAR_INDICES: Final[tuple[int, int]] = (LEFT_EAR, RIGHT_EAR)
SHOULDER_INDICES: Final[tuple[int, int]] = (LEFT_SHOULDER, RIGHT_SHOULDER)
ELBOW_INDICES: Final[tuple[int, int]] = (LEFT_ELBOW, RIGHT_ELBOW)
WRIST_INDICES: Final[tuple[int, int]] = (LEFT_WRIST, RIGHT_WRIST)

# (shoulder, elbow, wrist) per side, used to compute bilateral elbow angles.
SIDE_JOINTS: Final[dict[str, tuple[int, int, int]]] = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

KEYPOINT_NAMES: Final[dict[int, str]] = {
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

SKELETON_CONNECTIONS: Final[tuple[tuple[int, int], ...]] = (
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

# Keypoints rendered as skeleton joints (everything except the face).
SKELETON_KEYPOINTS: Final[tuple[int, ...]] = tuple(range(5, KEYPOINT_COUNT))


class PoseFrame(TypedDict):
    """One inference result for a single video frame.

    ``xy`` and ``conf`` are ``(n_people, n_keypoints, 2)`` and
    ``(n_people, n_keypoints)`` arrays respectively; ``person_conf`` is the
    per-person detection confidence ``(n_people,)``.
    """

    frame: int
    xy: np.ndarray
    conf: np.ndarray
    person_conf: np.ndarray

