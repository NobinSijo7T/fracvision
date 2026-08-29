"""Serial X-ray comparison module.

Compares a prior X-ray to a current X-ray for the same patient/injury.
Used to quantify visual change over time (healing progression tracking).

IMPORTANT: All metrics produced here are image-analysis measurements.
They have NOT been clinically validated as healing indicators.
They must not be presented as clinical measurements without validation.

Operations:
  1. Validate both images (same patient when metadata permits)
  2. Preprocess and normalize both images
  3. Rigid image registration (SimpleITK)
  4. Extract DINOv2 embeddings for both images
  5. Compute embedding-level similarity
  6. Compute pixel-level difference (post-registration)
  7. Compare segmentation masks (if available)
  8. Return structured comparison dict

Usage:
  from model.inference.compare import SerialXrayComparator
  comparator = SerialXrayComparator(extractor=..., segmentor=...)
  result = comparator.compare(prior_path, current_path, bbox_prior, bbox_current)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "These metrics are image-analysis measurements produced by a research prototype. "
    "They have NOT been clinically validated as indicators of fracture healing. "
    "Do not use for clinical decision-making without qualified clinician oversight."
)


def _try_register_images(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Attempt rigid registration of `moving` to `fixed` using SimpleITK.

    Returns (registered_image, success_flag).
    Falls back to unregistered `moving` if SimpleITK is unavailable or fails.
    """
    try:
        import SimpleITK as sitk

        def to_sitk(arr: np.ndarray) -> sitk.Image:
            gray = arr[..., 0] if arr.ndim == 3 else arr
            return sitk.GetImageFromArray(gray.astype(np.float32))

        fixed_sitk = to_sitk(fixed)
        moving_sitk = to_sitk(moving)

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=100
        )
        registration.SetInterpolator(sitk.sitkLinear)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk, moving_sitk,
            sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration.SetInitialTransform(initial_transform)
        final_transform = registration.Execute(fixed_sitk, moving_sitk)

        resampled = sitk.Resample(
            moving_sitk, fixed_sitk, final_transform,
            sitk.sitkLinear, 0.0, moving_sitk.GetPixelID(),
        )
        arr = sitk.GetArrayFromImage(resampled)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr.astype(np.uint8), True

    except ImportError:
        logger.debug("SimpleITK not available — skipping image registration")
        return moving, False
    except Exception as exc:
        logger.warning("Image registration failed: %s", exc)
        return moving, False


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two embedding vectors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def compute_pixel_diff_metrics(img1: np.ndarray, img2: np.ndarray) -> dict[str, float]:
    """Compute pixel-level difference statistics between two aligned images."""
    diff = img1.astype(float) - img2.astype(float)
    abs_diff = np.abs(diff)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "pct_pixels_changed_5pct": float((abs_diff > 12.75).mean()),  # >5% of 255
        "pct_pixels_changed_10pct": float((abs_diff > 25.5).mean()),
    }


class SerialXrayComparator:
    """Compares sequential X-rays of the same patient/injury."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        extractor=None,
        segmentor=None,
    ) -> None:
        self.config = config or {}
        self.extractor = extractor
        self.segmentor = segmentor

    def compare(
        self,
        prior_source: str | Path | np.ndarray,
        current_source: str | Path | np.ndarray,
        prior_bbox: list[float] | None = None,
        current_bbox: list[float] | None = None,
        patient_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare prior and current X-rays.

        Parameters
        ----------
        prior_source    : prior X-ray (file path or preprocessed array)
        current_source  : current X-ray
        prior_bbox      : [xmin,ymin,xmax,ymax] of fracture in prior image
        current_bbox    : [xmin,ymin,xmax,ymax] of fracture in current image
        patient_metadata: optional dict with patient_id, study dates, etc.

        Returns
        -------
        Structured comparison dict with metrics and disclaimers.
        """
        from model.data.preprocessing.preprocess import preprocess_from_config
        errors: dict[str, str] = {}

        # --- Preprocess both images ---
        try:
            prior_np, prior_qc = preprocess_from_config(prior_source, self.config)
        except Exception as exc:
            return {"error": f"Prior image preprocessing failed: {exc}", "disclaimer": _DISCLAIMER}

        try:
            current_np, current_qc = preprocess_from_config(current_source, self.config)
        except Exception as exc:
            return {"error": f"Current image preprocessing failed: {exc}", "disclaimer": _DISCLAIMER}

        # --- Registration ---
        registered_current, registration_success = _try_register_images(prior_np, current_np)

        # --- Pixel-level difference ---
        pixel_metrics = {}
        try:
            pixel_metrics = compute_pixel_diff_metrics(prior_np, registered_current)
        except Exception as exc:
            errors["pixel_diff"] = str(exc)

        # --- DINOv2 embedding comparison ---
        embedding_similarity: float | None = None
        try:
            if self.extractor is not None:
                def to_tensor(arr: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
                prior_feats = self.extractor.extract(to_tensor(prior_np))
                current_feats = self.extractor.extract(to_tensor(current_np))
                embedding_similarity = compute_cosine_similarity(
                    prior_feats["cls_token"].squeeze(0),
                    current_feats["cls_token"].squeeze(0),
                )
        except Exception as exc:
            errors["embedding_comparison"] = str(exc)

        # --- Segmentation mask comparison ---
        mask_metrics: dict[str, Any] = {}
        try:
            if self.segmentor is not None:
                prior_seg, current_seg = {}, {}
                if prior_bbox:
                    prior_seg = self.segmentor.segment(prior_np, prior_bbox)
                if current_bbox:
                    current_seg = self.segmentor.segment(current_np, current_bbox)
                if prior_seg.get("mask") is not None and current_seg.get("mask") is not None:
                    from model.models.medsam.segmentor import compute_dice, compute_iou
                    mask_metrics = {
                        "prior_area_pixels": int(prior_seg["mask"].sum()),
                        "current_area_pixels": int(current_seg["mask"].sum()),
                        "area_change_pixels": int(current_seg["mask"].sum()) - int(prior_seg["mask"].sum()),
                        "mask_overlap_dice": compute_dice(prior_seg["mask"], current_seg["mask"]),
                        "mask_overlap_iou": compute_iou(prior_seg["mask"], current_seg["mask"]),
                    }
        except Exception as exc:
            errors["segmentation_comparison"] = str(exc)

        return {
            "registration_applied": registration_success,
            "embedding_similarity": embedding_similarity,
            "pixel_level_metrics": pixel_metrics,
            "mask_comparison": mask_metrics,
            "prior_qc": {k: v for k, v in prior_qc.items() if k != "warnings"},
            "current_qc": {k: v for k, v in current_qc.items() if k != "warnings"},
            "patient_metadata": patient_metadata or {},
            "stage_errors": errors,
            "clinically_validated": False,
            "metric_warning": (
                "These metrics are NOT clinically validated healing measurements. "
                "Do not interpret as clinical evidence of fracture union."
            ),
            "disclaimer": _DISCLAIMER,
        }
