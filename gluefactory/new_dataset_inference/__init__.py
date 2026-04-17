"""Inference-only pipeline for the new_dataset ISS+FPFH/SHOT+LG stack.

Intended for GUI integration: single entry point that takes two raw zmaps
(uint16 PNG contents) and returns matched keypoints + a rigid transform.

Public API:
    from gluefactory.new_dataset_inference import (
        InferencePipeline, CameraCalib, ViewFeatures, InferenceResult,
        load_shot_bin_descriptors,
    )
"""
from .pipeline import (
    CameraCalib,
    InferencePipeline,
    InferenceResult,
    ViewFeatures,
    load_shot_bin_descriptors,
)

__all__ = [
    "CameraCalib",
    "InferencePipeline",
    "InferenceResult",
    "ViewFeatures",
    "load_shot_bin_descriptors",
]
