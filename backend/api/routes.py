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
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.core.pipeline_state import PipelineState
from backend.schemas.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    BoundingBoxSchema,
    SegmentationSchema,
    ClassificationSchema,
    ComparisonResponse,
    HistoryRecordSchema,
    CLINICAL_DISCLAIMER,
)
from backend.database.models import (
    ImagingStudy,
    Patient,
    Prediction,
    SessionLocal,
    get_db,
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
            mask_data_url=s.mask_data_url,
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

    response = PredictionResponse(
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

    # Persist the prediction without storing the uploaded image itself. The
    # de-identified upload is attached to one anonymous patient timeline until
    # patient identity support is added to the intake flow.
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == "anonymous-upload").first()
        if patient is None:
            patient = Patient(patient_id="anonymous-upload", dataset_source="UPLOAD")
            db.add(patient)
            db.flush()

        image_id = f"upload-{uuid4().hex}"
        study = ImagingStudy(
            patient_id=patient.patient_id,
            image_id=image_id,
            study_number=file.filename,
            fracture_visible=result.fracture_detected,
        )
        segmentation = result.segmentation
        prediction = Prediction(
            image_id=image_id,
            model_version="fracvision-local",
            fracture_detected=result.fracture_detected,
            fracture_confidence=result.fracture_confidence,
            detections_json=[d.__dict__ for d in result.detections],
            classification_json=(result.classification.__dict__ if result.classification else None),
            segmentation_json=(
                {
                    "mask_available": segmentation.mask_available,
                    "area_pixels": segmentation.area_pixels,
                    "mask_score": segmentation.mask_score,
                }
                if segmentation else None
            ),
            qc_passed=result.qc_passed,
            stage_errors_json=result.stage_errors,
        )
        db.add(study)
        db.add(prediction)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Prediction completed but history persistence failed: %s", exc)
    finally:
        db.close()

    return response


@router.get("/history", response_model=list[HistoryRecordSchema], tags=["History"])
async def history(db: Session = Depends(get_db)) -> list[HistoryRecordSchema]:
    """Return persisted upload predictions, newest first."""
    rows = (
        db.query(Prediction, ImagingStudy)
        .join(ImagingStudy, Prediction.image_id == ImagingStudy.image_id)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    records: list[HistoryRecordSchema] = []
    for prediction, study in rows:
        classification = prediction.classification_json or {}
        segmentation = prediction.segmentation_json or {}
        records.append(HistoryRecordSchema(
            id=prediction.id,
            image_id=prediction.image_id,
            filename=study.study_number,
            created_at=(prediction.created_at or datetime.utcnow()).isoformat(),
            fracture_detected=prediction.fracture_detected,
            fracture_confidence=float(prediction.fracture_confidence or 0.0),
            classification_labels=classification.get("labels", []),
            segmentation_available=bool(segmentation.get("mask_available", False)),
            mask_score=segmentation.get("mask_score"),
        ))
    return records


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
