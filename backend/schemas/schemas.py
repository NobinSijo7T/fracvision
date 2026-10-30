"""Pydantic schemas for FracVision API request/response models."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

CLINICAL_DISCLAIMER = (
    "This output is produced by a research prototype and has NOT been "
    "clinically validated. It must NOT be used for clinical decision-making, "
    "diagnosis, or treatment planning without supervision by a qualified clinician."
)


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    load_error: str | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = CLINICAL_DISCLAIMER


class BoundingBoxSchema(BaseModel):
    label: str
    bbox_xyxy: list[float] = Field(description="[xmin, ymin, xmax, ymax] in pixels")
    confidence: float = Field(ge=0.0, le=1.0)


class SegmentationSchema(BaseModel):
    mask_available: bool
    area_pixels: int
    mask_score: float | None
    model_loaded: bool
    mask_data_url: str | None = None
    warning: str | None = None


class ClassificationSchema(BaseModel):
    labels: list[str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated: bool


class PredictionResponse(BaseModel):
    image_path: str | None = None
    image_hw: list[int]
    qc_passed: bool
    qc_warnings: list[str]
    fracture_detected: bool
    fracture_confidence: float = Field(ge=0.0, le=1.0)
    detections: list[BoundingBoxSchema]
    segmentation: SegmentationSchema | None = None
    classification: ClassificationSchema | None = None
    stage_errors: dict[str, str] = Field(default_factory=dict)
    clinically_validated: bool = False
    recovery_training_enabled: bool = False
    disclaimer: str = CLINICAL_DISCLAIMER


class HistoryRecordSchema(BaseModel):
    id: int
    image_id: str
    filename: str | None = None
    created_at: str
    fracture_detected: bool | None = None
    fracture_confidence: float = 0.0
    classification_labels: list[str] = Field(default_factory=list)
    segmentation_available: bool = False
    mask_score: float | None = None


class ComparisonRequest(BaseModel):
    prior_image_id: str | None = None
    current_image_id: str | None = None
    patient_metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonResponse(BaseModel):
    registration_applied: bool
    embedding_similarity: float | None
    pixel_level_metrics: dict[str, float]
    mask_comparison: dict[str, Any]
    stage_errors: dict[str, str]
    clinically_validated: bool = False
    metric_warning: str
    disclaimer: str = CLINICAL_DISCLAIMER


class ModelInfoResponse(BaseModel):
    model_name: str
    embedding_dim: int | None = None
    class_names: list[str] | None = None
    checkpoint_loaded: bool
    device: str
