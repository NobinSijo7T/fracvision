"""DINOv2-based fracture classification head.

Trains a lightweight MLP classification head on top of frozen (or partially
fine-tuned) DINOv2 CLS token embeddings.

Supports:
  - Binary classification (fracture present/absent)
  - Multi-label classification (body part, view, etc.)
  - Focal loss for class imbalance
  - Temperature scaling for probability calibration

Only classes that actually exist in the dataset annotations are used.
No labels are invented.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for single or multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Temperature scaling calibration
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """Post-hoc probability calibration via a single learned temperature."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_iters: int = 200,
        lr: float = 0.01,
    ) -> None:
        """Fit temperature on validation logits/targets."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_iters)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        logger.info("Calibrated temperature: %.4f", self.temperature.item())


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class FractureClassifier(nn.Module):
    """MLP classification head on top of DINOv2 embeddings.

    Parameters
    ----------
    embedding_dim  : DINOv2 CLS token dimension (384/768/1024/1536)
    class_names    : list of class label names (determines output dim)
    hidden_dims    : hidden layer sizes (empty = linear probe)
    dropout        : dropout rate
    use_focal_loss : use focal loss instead of BCE
    focal_alpha    : focal loss alpha
    focal_gamma    : focal loss gamma
    """

    def __init__(
        self,
        embedding_dim: int,
        class_names: list[str],
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.embedding_dim = embedding_dim

        # Build MLP
        hidden_dims = hidden_dims or [512, 256]
        layers: list[nn.Module] = []
        in_dim = embedding_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, self.n_classes))
        self.head = nn.Sequential(*layers)

        # Loss
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Calibration
        self.calibrator = TemperatureScaling()
        self._is_calibrated = False

        logger.info(
            "FractureClassifier: %d classes, embedding_dim=%d, hidden=%s",
            self.n_classes, embedding_dim, hidden_dims,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits (B, n_classes)."""
        return self.head(embeddings)

    def predict(self, embeddings: torch.Tensor) -> dict[str, Any]:
        """Inference: returns calibrated probabilities and class predictions.

        Returns
        -------
        dict with:
            "logits"       : (B, n_classes) raw logits
            "probabilities": (B, n_classes) sigmoid probabilities
            "predictions"  : list[list[str]] — positive class names per sample
            "confidence"   : (B,) max probability across classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(embeddings)
            if self._is_calibrated:
                logits = self.calibrator(logits)
            probs = torch.sigmoid(logits)
            threshold = 0.5
            preds = []
            for i in range(probs.shape[0]):
                pos = [self.class_names[j] for j in range(self.n_classes) if probs[i, j] >= threshold]
                preds.append(pos)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "confidence": probs.max(dim=1).values,
        }

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal/BCE loss. targets shape: (B, n_classes), float."""
        logits = self.forward(embeddings)
        return self.criterion(logits, targets.float())

    def calibrate(self, val_embeddings: torch.Tensor, val_targets: torch.Tensor) -> None:
        """Calibrate temperature scaling on validation set."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(val_embeddings)
        self.calibrator.calibrate(logits, val_targets.float())
        self._is_calibrated = True

    def get_class_mapping(self) -> dict[int, str]:
        return {i: name for i, name in enumerate(self.class_names)}
