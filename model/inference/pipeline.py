"""Full FracVision inference pipeline.

Orchestrates all stages:
  1. Preprocess image
  2. RT-DETRv2 fracture detection
  3. MedSAM segmentation (for each detected fracture)
  4. DINOv2 embedding extraction
  5. Fracture classification
  6. (Optional) Longitudinal Mamba prediction

Every prediction includes confidence/uncertainty.
Every output carries explicit flags:
  clinically_validated: false
  recovery_training_enabled: false

Fails gracefully: individual stage errors are captured per-field.
The pipeline never crashes the entire application for a single bad image.

Usage:
  from model.inference.pipeline import FracVisionPipeline
  pipeline = FracVisionPipeline.from_config("model/configs/config.yaml")
  result = pipeline.predict("path/to/xray.jpg")
"""
from __future__ import annotations

import logging
import base64
from io import BytesIO
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.data.preprocessing.preprocess import preprocess_from_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    label: str
    bbox_xyxy: list[float]
    confidence: float


@dataclass
class SegmentationResult:
    mask_available: bool
    area_pixels: int
    mask_score: float | None
    model_loaded: bool
    mask_data_url: str | None = None
    warning: str | None = None


@dataclass
class ClassificationResult:
    labels: list[str]
    probabilities: dict[str, float]
    confidence: float
    calibrated: bool


@dataclass
class PredictionResult:
    # Input metadata
    image_path: str | None
    image_hw: list[int]
    qc_passed: bool
    qc_warnings: list[str]

    # Stage outputs
    fracture_detected: bool
    fracture_confidence: float
    detections: list[BoundingBox]
    segmentation: SegmentationResult | None
    classification: ClassificationResult | None

    # Mandatory disclaimers
    clinically_validated: bool = False
    recovery_training_enabled: bool = False
    disclaimer: str = (
        "This output is produced by a research prototype and has NOT been "
        "clinically validated. It must NOT be used for clinical decision-making, "
        "diagnosis, or treatment planning without supervision by a qualified clinician."
    )

    # Stage-level errors (populated when a stage fails)
    stage_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["detections"] = [asdict(b) for b in self.detections]
        d["segmentation"] = asdict(self.segmentation) if self.segmentation else None
        d["classification"] = asdict(self.classification) if self.classification else None
        return d


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FracVisionPipeline:
    """Full inference pipeline for FracVision.

    Parameters
    ----------
    config          : loaded config dict
    detector        : RTDETRv2Detector instance
    segmentor       : MedSAMSegmentor instance
    extractor       : DINOv2Extractor instance
    classifier      : FractureClassifier instance
    device          : torch device
    """

    def __init__(
        self,
        config: dict[str, Any],
        detector=None,
        segmentor=None,
        extractor=None,
        classifier=None,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.detector = detector
        self.segmentor = segmentor
        self.extractor = extractor
        self.classifier = classifier

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("FracVisionPipeline initialised on %s", self.device)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FracVisionPipeline":
        """Build pipeline from config file, loading all available models."""
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        models_cfg = cfg.get("models", {})
        infer_cfg = cfg.get("inference", {})
        pp_cfg = cfg.get("preprocessing", {})
        device = "auto"

        # DINOv2
        try:
            from model.models.dinov2 import DINOv2Extractor, FractureClassifier
            from model.training.train_classifier import FRACATLAS_LABELS
            dino_cfg = cfg.get("training", {}).get("phase_1_dinov2_adaptation", {})
            extractor = DINOv2Extractor(
                model_name=dino_cfg.get("model_name", "dinov2_vits14"),
                checkpoint_path=models_cfg.get("dinov2_checkpoint"),
                device=device,
            )
            classifier = FractureClassifier(
                embedding_dim=extractor.embedding_dim,
                class_names=FRACATLAS_LABELS,
            )
            ckpt = models_cfg.get("dinov2_classifier_best")
            if ckpt and Path(ckpt).exists():
                state = torch.load(ckpt, map_location="cpu")
                classifier.load_state_dict(state, strict=False)
                logger.info("Loaded classifier checkpoint: %s", ckpt)
            # DINOv2 places embeddings on its resolved device.  The classifier
            # is created on CPU by default, so explicitly colocate it with the
            # extractor before inference (especially when CUDA is available).
            classifier = classifier.to(extractor.device)
            classifier.eval()
        except Exception as exc:
            logger.warning("DINOv2/Classifier load failed: %s", exc)
            extractor, classifier = None, None

        # RT-DETRv2
        try:
            from model.models.rtdetrv2.detector import RTDETRv2Detector
            all_classes = list(set(
                cfg.get("class_mappings", {}).get("grazpedwri", {}).get("detection_classes", ["fracture"])
            ))
            detector = RTDETRv2Detector(
                class_names=sorted(all_classes),
                score_threshold=infer_cfg.get("detection_score_threshold", 0.30),
                checkpoint_path=models_cfg.get("rtdetrv2_checkpoint"),
                device=device,
            )
        except Exception as exc:
            logger.warning("RTDETRv2 load failed: %s", exc)
            detector = None

        # MedSAM
        try:
            from model.models.medsam.segmentor import MedSAMSegmentor
            seg_cfg = cfg.get("training", {}).get("phase_3_segmentation", {})
            segmentor = MedSAMSegmentor(
                checkpoint_path=models_cfg.get("medsam_checkpoint"),
                device=device,
                dice_weight=seg_cfg.get("dice_weight", 0.5),
                bce_weight=seg_cfg.get("bce_weight", 0.5),
            )
        except Exception as exc:
            logger.warning("MedSAM load failed: %s", exc)
            segmentor = None

        return cls(cfg, detector, segmentor, extractor, classifier, device=device)

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(self, source: str | Path | bytes | np.ndarray) -> PredictionResult:
        """Run full inference on a single X-ray.

        Handles invalid/corrupt images gracefully — returns partial results
        with populated stage_errors rather than raising exceptions.
        """
        stage_errors: dict[str, str] = {}
        image_path_str = str(source) if isinstance(source, Path) else (
            str(source) if isinstance(source, str) else None
        )

        # --- Stage 1: Preprocess ---
        try:
            image_np, qc_meta = preprocess_from_config(source, self.config, return_tensor=False)
            image_hw = list(qc_meta.get("output_hw", [0, 0]))
            qc_passed = qc_meta.get("qc_passed", False)
            qc_warnings = qc_meta.get("warnings", [])
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Preprocessing failed: %s", exc)
            return PredictionResult(
                image_path=image_path_str,
                image_hw=[0, 0],
                qc_passed=False,
                qc_warnings=[str(exc)],
                fracture_detected=False,
                fracture_confidence=0.0,
                detections=[],
                segmentation=None,
                classification=None,
                stage_errors={"preprocessing": str(exc)},
            )

        # --- Stage 2: Detection ---
        detections: list[BoundingBox] = []
        fracture_detected = False
        fracture_confidence = 0.0
        primary_bbox: list[float] | None = None

        try:
            if self.detector is not None:
                det_result = self.detector.detect(image_np)
                fracture_detected = det_result.fracture_detected
                fracture_confidence = det_result.fracture_confidence
                for box, score, label in zip(det_result.boxes, det_result.scores, det_result.labels):
                    detections.append(BoundingBox(label=label, bbox_xyxy=box, confidence=score))
                if detections:
                    # Use highest-confidence fracture box as MedSAM prompt
                    frac_boxes = [d for d in detections if d.label == "fracture"]
                    if frac_boxes:
                        primary_bbox = max(frac_boxes, key=lambda d: d.confidence).bbox_xyxy
        except Exception as exc:
            logger.error("Detection stage failed: %s", exc)
            stage_errors["detection"] = str(exc)

        # --- Stage 3: Segmentation ---
        seg_result: SegmentationResult | None = None
        try:
            if self.segmentor is not None and primary_bbox is not None:
                seg_out = self.segmentor.segment(image_np, primary_bbox)
                mask_data_url = None
                mask = seg_out.get("mask")
                if mask is not None and np.asarray(mask).any():
                    from PIL import Image
                    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                    mask_rgba[..., 0] = 248
                    mask_rgba[..., 1] = 113
                    mask_rgba[..., 2] = 113
                    mask_rgba[..., 3] = np.asarray(mask, dtype=bool).astype(np.uint8) * 120
                    buffer = BytesIO()
                    Image.fromarray(mask_rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
                    mask_data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                seg_result = SegmentationResult(
                    mask_available=seg_out.get("model_loaded", False),
                    area_pixels=seg_out.get("area_pixels", 0),
                    mask_score=seg_out.get("mask_score"),
                    model_loaded=seg_out.get("model_loaded", False),
                    mask_data_url=mask_data_url,
                    warning=seg_out.get("warning"),
                )
        except Exception as exc:
            logger.error("Segmentation stage failed: %s", exc)
            stage_errors["segmentation"] = str(exc)

        # --- Stage 4: Embedding + Classification ---
        cls_result: ClassificationResult | None = None
        try:
            if self.extractor is not None and self.classifier is not None:
                img_t = torch.from_numpy(image_np.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
                feats = self.extractor.extract(img_t)
                cls_out = self.classifier.predict(feats["cls_token"])
                probs_dict = {
                    name: float(cls_out["probabilities"][0, i])
                    for i, name in enumerate(self.classifier.class_names)
                }
                cls_result = ClassificationResult(
                    labels=cls_out["predictions"][0],
                    probabilities=probs_dict,
                    confidence=float(cls_out["confidence"][0]),
                    calibrated=self.classifier._is_calibrated,
                )
        except Exception as exc:
            logger.error("Classification stage failed: %s", exc)
            stage_errors["classification"] = str(exc)

        return PredictionResult(
            image_path=image_path_str,
            image_hw=image_hw,
            qc_passed=qc_passed,
            qc_warnings=qc_warnings,
            fracture_detected=fracture_detected,
            fracture_confidence=float(fracture_confidence),
            detections=detections,
            segmentation=seg_result,
            classification=cls_result,
            stage_errors=stage_errors,
        )
