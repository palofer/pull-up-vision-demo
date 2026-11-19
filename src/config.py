"""Shared configuration thresholds for pose analysis and visualization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    keypoint_confidence: float = 0.5
    clearance_confidence: float = 0.3
    clearance_threshold_px: float = 20.0
    extension_threshold_deg: float = 150.0
    initiation_threshold_deg: float = 130.0
    rom_extension_reference_deg: float = 180.0
    rom_flexion_reference_deg: float = 55.0
    black_bar_threshold: float = 12.0
    crop_sample_count: int = 7


@dataclass(frozen=True)
class VisualConfig:
    """Layout constants for rendered artifacts, separated from analysis thresholds."""

    chart_margin_px: float = 8.0
    chart_margin_ratio: float = 0.1
    label_offset_ratio: float = 0.08
    dashboard_height_px: int = 1024
    dashboard_panel_width_px: int = 680
    dashboard_chart_top_px: int = 92
    table_header_gap_px: int = 28
    table_row_height_px: int = 30
    table_max_rows: int = 10
    table_column_xs: tuple[int, ...] = (30, 95, 245, 355, 455, 565)


DEFAULT_CONFIG = AnalysisConfig()
DEFAULT_VISUAL_CONFIG = VisualConfig()
