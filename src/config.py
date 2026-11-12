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


DEFAULT_CONFIG = AnalysisConfig()
