# Pull-Up Vision

Local computer vision pipeline that analyzes pull-up video clips using **YOLO11-pose** (`COCO-17` keypoints). It counts repetitions, times concentric/eccentric phases, computes bilateral elbow angles and range of motion (ROM), estimates ear clearance relative to the grip, and exports both an interactive HTML viewer (single file; the source video is referenced by relative path) and an optional split-screen H.264 MP4 dashboard.

> **Disclaimer:** This project is an experimental computer vision prototype and an engineering study on monocular 2D pose estimation limits. It is not a medical assessment or a clinical biomechanics tool.

![Pull-Up Vision Dashboard Preview](readme_images/dashboard-preview.png)

---

## Quick Start

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (plus `ffmpeg` on `PATH` if exporting H.264 MP4 videos).

1. **Install dependencies:**

```bash
uv sync
```

2. **Add a pull-up video** to `data/raw/` (filmed from the front or back with the full body and bar visible in frame).

3. **Run the pipeline:**

```bash
# Run on the first video found in data/raw/
uv run python src/main.py

# Run on a specific clip
uv run python src/main.py data/raw/my-set.mp4

# Also render a split-screen H.264 MP4 dashboard video
uv run python src/main.py data/raw/my-set.mp4 --export-mp4
```

On the first run, Ultralytics automatically downloads `yolo11m-pose.pt` (ignored by Git).

| Flag / Argument | Description |
| :--- | :--- |
| `VIDEO` *(optional)* | Path to input video clip (defaults to the first clip in `data/raw/`) |
| `--weights PATH` | Pose model weights path or identifier (default: `yolo11m-pose.pt`) |
| `--export-mp4` | Render a synchronized split-screen H.264 MP4 video alongside the HTML viewer |

All analysis thresholds are centralized in [`src/config.py`](src/config.py).

---

## Output Artifacts

Each analyzed clip produces a directory under `data/processed/<clip_name>/`:

```text
data/processed/<clip_name>/
├── viewer.html                # Interactive HTML5 viewer (skeleton + live charts over the source video)
├── <clip_name>-dashboard.mp4  # Optional (--export-mp4) split-screen H.264 video with audio
├── plots.png                  # High-resolution 3-panel static summary chart
├── elbow-angles.png           # Bilateral elbow angle trajectory chart
├── shoulder-position.png      # Vertical shoulder trajectory and |Δy| tilt chart
├── summary.json               # Structured run metadata, config, aggregate stats, and per-rep breakdown
├── summary.csv                # Single-row aggregate metrics (counts, mean ROM %, mean ascent/descent)
├── repetitions.csv            # Per-repetition frame bounds, min angles, ear clearance, ROM %, and timings
├── elbow-angles.csv           # Per-frame bilateral elbow angles and minimum keypoint confidences
└── keypoints.csv              # Raw per-frame COCO-17 keypoint coordinates and confidences
```

Open the interactive viewer on macOS:

```bash
open data/processed/<clip_name>/viewer.html
```

---

## How It Works

Code architecture:
- [`src/main.py`](src/main.py) orchestrates inference, CLI parsing, and exports.
- [`src/angles.py`](src/angles.py) computes joint angles (`arctan2`), grip clearance, repetition state transitions, and ROM/tempo metrics.
- [`src/pose.py`](src/pose.py) holds the COCO-17 keypoint topology and naming shared across modules.
- [`src/viewer.py`](src/viewer.py) & [`src/viewer-template.html`](src/viewer-template.html) build the zero-dependency interactive HTML viewer (with automatic lateral pillarbox cropping).
- [`src/visual.py`](src/visual.py) renders the static diagnostic plots (matplotlib PNGs).
- [`src/render.py`](src/render.py) renders the split-screen H.264 MP4 dashboard video.
- [`src/config.py`](src/config.py) holds all confidence and geometric thresholds, plus layout constants.

### 1. Repetition Detection & Grading
- **Cycle Boundaries (Elbow State Machine):** A repetition arms when both elbows extend past `extension_threshold_deg` (`150°`), initiates once either elbow flexes below `initiation_threshold_deg` (`130°`), tracks peak flexion (`top`), and closes when both elbows return to extension (`≥ 150°`).
- **Top Clearance (Ear Height vs. Grip):** Because COCO-17 lacks a chin keypoint, ear height (`left_ear`, `right_ear`) is measured relative to the fixed grip reference (median vertical coordinate of valid `left_wrist` and `right_wrist` detections).
- **Repetition States:**
  - `complete`: Cycle returns to full extension (`≥ 150°`) and ears clear the grip reference by at least `+25 px`.
  - `partial`: Cycle returns to extension (`≥ 150°`) but fails to reach the `+25 px` ear clearance threshold.
  - `no_extension`: Athlete initiates a pull, but the clip ends (or the athlete drops off the bar) before returning to elbow extension. Tracked separately from partial reps.

---

## Empirical Findings: Limitations of Monocular 2D Pose Estimation

A central goal of this project was to evaluate whether an off-the-shelf monocular 2D pose detector (`COCO-17` keypoints) can reliably detect **subtle biomechanical technique faults**—specifically **bilateral pulling asymmetry** (one shoulder/arm dominating the concentric phase) and **mid-pull compensatory pauses**—beyond basic repetition counting and macro ROM grading.

By evaluating the pipeline across multiple manually annotated sets (`34` labeled repetitions across `6` recordings covering clean sets, pause sets, and intentional asymmetry — labels in [`data/labels/manual-repetition-labels.csv`](data/labels/manual-repetition-labels.csv)), we reached a clear empirical conclusion: **2D keypoint jitter and minor camera perspective variation exceed the signal magnitude of subtle bilateral asymmetry.**

### Quantitative Comparison: Real Asymmetry vs. Clean Sets

| Dataset & Repetition | Ground-Truth Annotation | Shoulder Vertical Tilt `\|Δy\|` Mean / Max (px) | Normalized Shoulder Tilt (`\|Δy\|` / Shoulder Width) | Peak Flexion Elbow Diff `\|L - R\|` (°) | Bilateral ROM Diff `\|ROM_L - ROM_R\|` (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Athlete A — Set 1, Rep 2** | **`asymmetry = present`** | **0.9 px / 2.4 px** *(0.36 px at peak)* | **1.2% / 3.3%** | **6.5°** *(73.5° vs 67.0°)* | **10.9%** *(62.0% vs 72.9%)* |
| **Athlete B — Set 1, Rep 2** | **`technique = perfect`** | 1.7 px / 4.2 px | 1.8% / 4.6% | **9.5°** *(52.2° vs 61.7°)* | **13.8%** *(86.4% vs 72.6%)* |
| **Athlete B — Set 1, Rep 8** | **`technique = perfect`** | **3.2 px / 6.3 px** *(3.29 px at peak)* | **3.4% / 6.8%** | 3.9° *(54.2° vs 50.3°)* | 5.1% *(85.7% vs 80.6%)* |
| **Athlete C — Set 1, Rep 1** | **`technique = perfect`** | **3.3 px / 5.6 px** *(3.42 px at peak)* | **3.1% / 5.4%** | 2.0° *(44.5° vs 46.5°)* | 3.1% *(90.6% vs 87.5%)* |
| **Athlete C — Set 1, Rep 2** | **`technique = perfect`** | 1.8 px / 4.3 px | 1.7% / 4.1% | **9.7°** *(42.1° vs 51.8°)* | 0.1% *(93.8% vs 93.9%)* |
| **Athlete A — Set 2, Rep 1** | `perfect_with_pause` | 0.9 px / 2.3 px | 1.7% / 4.3% | **9.5°** *(31.9° vs 22.4°)* | **8.1%** *(97.3% vs 105.4%)* |

### Why 2D COCO-17 Fails as a Strict Biomechanical Asymmetry Judge

1. **Shoulder Vertical Tilt (`|y_left - y_right|`) is Dominated by Camera Roll & Deltoid Keypoint Drift:**
   - In the ground-truth asymmetric repetition (*Athlete A, Set 1, Rep 2*), mean vertical shoulder tilt is **0.9 px** (**1.2%** of biacromial shoulder width).
   - Across 15 repetitions manually graded as `perfect` (*Athlete B* and *Athlete C*), normalized shoulder tilt reaches **1.6%–3.4% mean** (up to **6.8% max**) simply due to a `1°–2°` handheld/tripod roll angle and soft-tissue keypoint jitter over the posterior deltoid during scapular retraction.
2. **2D Foreshortening Corrupts Bilateral Elbow Angles (`|left_elbow_deg - right_elbow_deg|`):**
   - Projecting a 3D hinge joint onto a single 2D image plane makes the apparent angle highly sensitive to internal/external shoulder rotation and camera yaw.
   - Clean repetitions (*Athlete B Rep 2*, *Athlete C Rep 2*) exhibit **9.5°–9.7°** bilateral elbow differences at peak flexion and up to **13.8%** bilateral ROM divergence—exceeding the **6.5°** / **10.9%** measured during true asymmetric pulling.
3. **Lack of Scapular & 3D Depth Landmarks:**
   - Standard `COCO-17` topology only provides two shoulder centroids (`5`, `6`), omitting acromion, inferior scapular angle, and spine landmarks required to distinguish true scapular elevation/winging from camera tilt.

**Takeaway:** Monocular 2D YOLO11-pose is effective for **macro repetition counting, tempo estimation, and coarse top/bottom ROM grading**, which is where this repository focuses its interactive tooling. Detecting fine-grained technique compensations reliably requires either calibrated multi-view 3D pose estimation, depth sensors, or specialized anatomical keypoint topologies.

---

## Development & Testing

Run the unit and integration test suite and linter:

```bash
uv run --group dev python -m pytest
uv run --group dev ruff check src tests
```

Raw videos (`data/raw/`), processed outputs (`data/processed/`), and model weights (`*.pt`) are gitignored.

---

## Inspiration & Credits

Inspired by Jeremy Park's pull-up analysis demo in [`jeremyipark/vision-demos/tree/main/pull_ups`](https://github.com/jeremyipark/vision-demos/tree/main/pull_ups). Built from scratch as an independent local pipeline using Ultralytics YOLO11-pose, custom state-machine grading, and a zero-dependency HTML5 Canvas viewer.

## License

[MIT](LICENSE).
