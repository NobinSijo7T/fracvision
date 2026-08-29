"""DINOv2 feature extractor for X-ray representation learning.

Loads pretrained DINOv2 from torch.hub (auto-downloads on first run).
Supports frozen-feature mode (default) and optional partial fine-tuning
of the top N transformer blocks.

Pretrained weights are NOT trained from scratch — only adaptation is applied.

Usage:
  extractor = DINOv2Extractor(model_name="dinov2_vits14")
  features = extractor.extract(image_tensor)   # returns dict
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Available DINOv2 model names (from facebookresearch/dinov2)
DINOV2_MODELS = {
    "dinov2_vits14": 384,   # Small  — 21M params, embedding dim 384
    "dinov2_vitb14": 768,   # Base   — 86M params, embedding dim 768
    "dinov2_vitl14": 1024,  # Large  — 307M params, embedding dim 1024
    "dinov2_vitg14": 1536,  # Giant  — 1.1B params, embedding dim 1536
}


class DINOv2Extractor(nn.Module):
    """Pretrained DINOv2 feature extractor.

    Parameters
    ----------
    model_name     : DINOv2 variant name (see DINOV2_MODELS)
    fine_tune_layers : number of transformer blocks to unfreeze (0 = fully frozen)
    checkpoint_path  : optional local checkpoint to load instead of torch.hub
    device         : "cuda", "cpu", or "auto"
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        fine_tune_layers: int = 0,
        checkpoint_path: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__()

        if model_name not in DINOV2_MODELS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Choose from: {list(DINOV2_MODELS)}"
            )

        self.model_name = model_name
        self.embedding_dim = DINOV2_MODELS[model_name]
        self.fine_tune_layers = fine_tune_layers

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("DINOv2Extractor: using device %s", self.device)

        # Load model
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info("Loading DINOv2 from local checkpoint: %s", checkpoint_path)
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=False
            )
            state = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)
        else:
            if checkpoint_path:
                logger.warning(
                    "DINOv2 checkpoint not found at %s — downloading from torch.hub",
                    checkpoint_path,
                )
            logger.info("Loading pretrained DINOv2 (%s) from torch.hub ...", model_name)
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True
            )

        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze top N transformer blocks
        if fine_tune_layers > 0:
            blocks = list(self.backbone.blocks)
            for block in blocks[-fine_tune_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Always unfreeze norm layer
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(
                "DINOv2: unfroze top %d blocks → %d trainable parameters",
                fine_tune_layers, n_trainable,
            )
        else:
            logger.info("DINOv2: fully frozen (frozen-feature baseline mode)")

        self.backbone = self.backbone.to(self.device)

    # ------------------------------------------------------------------
    # Forward / extraction
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract DINOv2 features.

        Parameters
        ----------
        x : torch.Tensor — shape (B, 3, H, W), float32, values in [0,1]

        Returns
        -------
        dict with:
            "cls_token"    : (B, D) — global image representation
            "patch_tokens" : (B, N_patches, D) — spatial patch features
            "embedding"    : (B, D) — alias for cls_token (for compatibility)
        """
        x = x.to(self.device)
        with torch.set_grad_enabled(self.training and self.fine_tune_layers > 0):
            output = self.backbone.forward_features(x)

        cls_token = output["x_norm_clstoken"]           # (B, D)
        patch_tokens = output["x_norm_patchtokens"]      # (B, N, D)

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "embedding": cls_token,  # convenience alias
        }

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-only extraction (always no-grad)."""
        self.eval()
        return self.forward(x)

    @torch.no_grad()
    def extract_batch(
        self,
        images: list[torch.Tensor],
        batch_size: int = 8,
    ) -> list[dict[str, torch.Tensor]]:
        """Extract features from a list of image tensors in batches."""
        self.eval()
        results = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i : i + batch_size]).to(self.device)
            out = self.forward(batch)
            for j in range(batch.shape[0]):
                results.append({k: v[j] for k, v in out.items()})
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "fine_tune_layers": self.fine_tune_layers,
            "device": str(self.device),
        }
