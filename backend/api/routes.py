"""FastAPI route handlers for FracVision.

Endpoints:
  GET  /api/health          — service health + model status
  GET  /api/models          — model metadata
  POST /api/predict         — single image prediction
  POST /api/compare         — two-image serial comparison
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.core.pipeline_state import PipelineState
from backend.schemas.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    BoundingBoxSchema,
    SegmentationSchema,
    ClassificationSchema,
    ComparisonResponse,
    CLINICAL_DISCLAIMER,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Status"])
async def health() -> HealthResponse:
    """Return service health and model loading status."""
    model_versions: dict[str, str] = {}
    if PipelineState.pipeline is not None:
        p = PipelineState.pipeline
        if p.extractor:
            model_versions["dinov2"] = p.extractor.model_name
        if p.detector:
            model_versions["rtdetrv2"] = "rtdetrv2_r50vd"
        if p.segmentor:
            model_versions["medsam"] = "loaded" if p.segmentor._model_loaded else "checkpoint_missing"

    return HealthResponse(
        status="ok" if PipelineState.is_ready() else "degraded",
        pipeline_ready=PipelineState.is_ready(),
        load_error=PipelineState.load_error,
        model_versions=model_versions,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=list[ModelInfoResponse], tags=["Status"])
async def list_models() -> list[ModelInfoResponse]:
    """Return metadata for all loaded model components."""
    models: list[ModelInfoResponse] = []
    p = PipelineState.pipeline

    if p is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    if p.extractor:
        models.append(ModelInfoResponse(
            model_name=p.extractor.model_name,
            embedding_dim=p.extractor.embedding_dim,
            checkpoint_loaded=True,
            device=str(p.extractor.device),
        ))
    if p.detector:
        models.append(ModelInfoResponse(
            model_name="RTDETRv2",
            class_names=p.detector.class_names,
            checkpoint_loaded=True,
            device=str(p.detector.device),
        ))
    if p.segmentor:
        models.append(ModelInfoResponse(
            model_name="MedSAM",
            checkpoint_loaded=p.segmentor._model_loaded,
            device=str(p.segmentor.device),
        ))
    return models


# ---------------------------------------------------------------------------
# Predict (single image)
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Upload a single X-ray image and receive fracture analysis.

    Accepted formats: JPEG, PNG, DICOM (.dcm).
    Max file size: 50 MB.
    """
    if not PipelineState.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not ready: {PipelineState.load_error or 'unknown error'}",
        )

    # Validate file size
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {_MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )

    # Write to temp file (preserving extension for DICOM detection)
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".dcm"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix}. Use JPEG, PNG, or DICOM.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = PipelineState.pipeline.predict(tmp_path)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Map to response schema
    detections = [
        BoundingBoxSchema(label=d.label, bbox_xyxy=d.bbox_xyxy, confidence=d.confidence)
        for d in result.detections
    ]
    seg = None
    if result.segmentation:
        s = result.segmentation
        seg = SegmentationSchema(
            mask_available=s.mask_available,
            area_pixels=s.area_pixels,
            mask_score=s.mask_score,
            model_loaded=s.model_loaded,
            warning=s.warning,
        )
    cls_out = None
    if result.classification:
        c = result.classification
        cls_out = ClassificationSchema(
            labels=c.labels,
            probabilities=c.probabilities,
            confidence=c.confidence,
            calibrated=c.calibrated,
        )

    return PredictionResponse(
        image_path=file.filename,
        image_hw=result.image_hw,
        qc_passed=result.qc_passed,
        qc_warnings=result.qc_warnings,
        fracture_detected=result.fracture_detected,
        fracture_confidence=result.fracture_confidence,
        detections=detections,
        segmentation=seg,
        classification=cls_out,
        stage_errors=result.stage_errors,
    )


# ---------------------------------------------------------------------------
# Compare (serial X-rays)
# ---------------------------------------------------------------------------

@router.post("/compare", response_model=ComparisonResponse, tags=["Inference"])
async def compare(
    prior: UploadFile = File(...),
    current: UploadFile = File(...),
) -> ComparisonResponse:
    """Upload prior and current X-rays to compare them.

    Returns image registration, embedding similarity, pixel-level diff,
    and mask overlap metrics. All metrics are clearly labeled as
    NOT clinically validated.
    """
    if not PipelineState.is_ready():
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    def read_to_tmp(upload: UploadFile) -> str:
        suffix = Path(upload.filename or "upload.jpg").suffix.lower() or ".jpg"
        import asyncio
        return suffix

    prior_contents = await prior.read()
    current_contents = await current.read()

    for label, contents in [("prior", prior_contents), ("current", current_contents)]:
        if len(contents) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{label} file too large (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB)",
            )

    suffix_prior = Path(prior.filename or "prior.jpg").suffix.lower() or ".jpg"
    suffix_curr = Path(current.filename or "current.jpg").suffix.lower() or ".jpg"

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_prior) as f1:
        f1.write(prior_contents); prior_path = f1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_curr) as f2:
        f2.write(current_contents); current_path = f2.name

    try:
        from model.inference.compare import SerialXrayComparator
        p = PipelineState.pipeline
        comparator = SerialXrayComparator(
            config=p.config,
            extractor=p.extractor,
            segmentor=p.segmentor,
        )
        result = comparator.compare(prior_path, current_path)
    except Exception as exc:
        logger.error("Comparison error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")
    finally:
        Path(prior_path).unlink(missing_ok=True)
        Path(current_path).unlink(missing_ok=True)

    return ComparisonResponse(
        registration_applied=result.get("registration_applied", False),
        embedding_similarity=result.get("embedding_similarity"),
        pixel_level_metrics=result.get("pixel_level_metrics", {}),
        mask_comparison=result.get("mask_comparison", {}),
        stage_errors=result.get("stage_errors", {}),
        metric_warning=result.get("metric_warning", ""),
    )
