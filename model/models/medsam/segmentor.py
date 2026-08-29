"""MedSAM segmentation wrapper.

MedSAM (2024) is a medical image Segment Anything Model fine-tuned on
over 1 million medical image-mask pairs. It accepts bounding-box prompts
and generates pixel-level segmentation masks.

This wrapper:
  - Loads a MedSAM checkpoint if available (warns and skips gracefully if not)
  - Accepts preprocessed X-ray images + RT-DETRv2 bounding boxes as prompts
  - Returns binary segmentation masks with Dice/IoU metrics
  - Computes Dice + BCE loss when ground-truth masks are provided (FracAtlas COCO)

Note: MedSAM checkpoint must be downloaded separately.
  See: https://github.com/bowang-lab/MedSAM
  Place checkpoint at the path specified in config.yaml:
    models.medsam_checkpoint: model/checkpoints/medsam.pth
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).view(-1)
        target = target.float().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float())
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()
    intersection = (pred & gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return float(2 * intersection / (pred.sum() + gt.sum()))


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# MedSAM wrapper
# ---------------------------------------------------------------------------

class MedSAMSegmentor(nn.Module):
    """MedSAM-based fracture region segmentor.

    Parameters
    ----------
    checkpoint_path : path to MedSAM .pth checkpoint
    device          : "auto", "cuda", or "cpu"
    dice_weight     : weight for Dice loss component
    bce_weight      : weight for BCE loss component
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "auto",
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        self.criterion = DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
        self._model_loaded = False
        self._sam_model = None
        self._sam_predictor = None

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Attempt to load MedSAM. Warns and continues if unavailable."""
        if not self.checkpoint_path:
            logger.warning(
                "MedSAM checkpoint path not configured. "
                "Segmentation will return placeholder masks. "
                "Set models.medsam_checkpoint in config.yaml."
            )
            return

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(
                "MedSAM checkpoint not found at %s. "
                "Segmentation will return placeholder masks. "
                "Download from: https://github.com/bowang-lab/MedSAM",
                ckpt_path,
            )
            return

        try:
            # Try to load via segment_anything
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_path))
            sam = sam.to(self.device)
            self._sam_model = sam
            self._sam_predictor = SamPredictor(sam)
            self._model_loaded = True
            logger.info("MedSAM loaded from %s on %s", ckpt_path, self.device)
        except ImportError:
            logger.warning(
                "segment_anything package not found. "
                "Install with: pip install segment-anything"
            )
        except Exception as exc:
            logger.error("Failed to load MedSAM checkpoint: %s", exc)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def segment(
        self,
        image_np: np.ndarray,
        bbox_xyxy: list[float],
    ) -> dict[str, Any]:
        """Segment a region given an image and a bounding-box prompt.

        Parameters
        ----------
        image_np  : np.ndarray (H, W, 3) uint8
        bbox_xyxy : [xmin, ymin, xmax, ymax] absolute pixel coords

        Returns
        -------
        dict:
            mask         : np.ndarray (H, W) bool
            mask_score   : float confidence (None if not available)
            area_pixels  : int
            model_loaded : bool — whether real MedSAM was used
        """
        h, w = image_np.shape[:2]

        if not self._model_loaded or self._sam_predictor is None:
            # Return empty mask with clear label
            empty_mask = np.zeros((h, w), dtype=bool)
            return {
                "mask": empty_mask,
                "mask_score": None,
                "area_pixels": 0,
                "model_loaded": False,
                "warning": (
                    "MedSAM checkpoint not available. "
                    "This is a placeholder mask — not a clinical segmentation."
                ),
            }

        try:
            self._sam_predictor.set_image(image_np)
            box = np.array(bbox_xyxy, dtype=np.float32)
            masks, scores, _ = self._sam_predictor.predict(
                box=box[None, :],
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
            score = float(scores[0]) if scores is not None else None
            return {
                "mask": mask,
                "mask_score": score,
                "area_pixels": int(mask.sum()),
                "model_loaded": True,
            }
        except Exception as exc:
            logger.error("MedSAM segmentation failed: %s", exc)
            return {
                "mask": np.zeros((h, w), dtype=bool),
                "mask_score": None,
                "area_pixels": 0,
                "model_loaded": True,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Training helpers (when GT masks are available from FracAtlas COCO)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice + BCE loss for supervised segmentation training."""
        return self.criterion(pred_logits, gt_mask)

    def evaluate_masks(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, float]:
        """Compute Dice and IoU between predicted and ground-truth masks."""
        return {
            "dice": compute_dice(pred_mask, gt_mask),
            "iou": compute_iou(pred_mask, gt_mask),
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "model_loaded": self._model_loaded,
            "device": str(self.device),
        }
