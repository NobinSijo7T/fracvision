"""Mamba-based longitudinal feature model for sequential X-ray analysis.

Mamba (2023) uses selective state space models (SSMs) to efficiently model
long-range dependencies in sequences — superior to transformers for variable-
length time series with irregular sampling.

Architecture:
  Input sequence: [DINOv2_embedding || detection_features || seg_features ||
                   time_delta || clinical_features] for each visit
  Output: per-timestep healing-stage probability + global sequence representation

Two implementations are provided:
  1. mamba-ssm (CUDA-optimised, requires CUDA + mamba-ssm package)
  2. Pure-PyTorch Mamba fallback (slower but CPU-compatible)

Recovery-time supervised training is DISABLED (no clinical outcome labels).
Only healing-stage ordinal classification (fracture_visible label progression)
is trained when GRAZPEDWRI-DX fracture_visible labels are available.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba block (fallback when mamba-ssm is not available)
# ---------------------------------------------------------------------------

class MambaBlockPyTorch(nn.Module):
    """Single Mamba-style selective SSM block (pure PyTorch implementation).

    This is a simplified Mamba block for CPU compatibility. For full
    performance on GPU, install mamba-ssm and the CUDA version is used.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
        Returns : (B, L, D)
        """
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        xz = self.in_proj(x)           # (B, L, 2 * d_inner)
        x_p, z = xz.chunk(2, dim=-1)   # each (B, L, d_inner)

        # Causal conv
        x_p = x_p.transpose(1, 2)      # (B, d_inner, L)
        x_p = self.conv1d(x_p)[..., :L]
        x_p = x_p.transpose(1, 2)      # (B, L, d_inner)
        x_p = F.silu(x_p)

        # SSM parameters (simplified linear scan for CPU)
        A = -torch.exp(self.A_log.float())   # (d_inner, d_state)
        x_dbl = self.x_proj(x_p)             # (B, L, d_state*2 + d_model)
        B_proj, C, dt_raw = x_dbl.split([self.d_state, self.d_state, D], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # Simplified sequential scan (for correctness, not speed)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)          # (B, d_inner, 1)
            A_dt = torch.exp(A.unsqueeze(0) * dt_t)   # (B, d_inner, d_state)
            B_t = B_proj[:, t, :].unsqueeze(1)        # (B, 1, d_state)
            h = A_dt * h + dt_t * B_t * x_p[:, t, :].unsqueeze(-1)
            C_t = C[:, t, :].unsqueeze(-1)            # (B, d_state, 1)
            y = (h * C_t.transpose(1, 2)).sum(dim=-1) # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)     # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_p

        # Gate
        y = y * F.silu(z)
        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# Try to import CUDA Mamba SSM (optional)
# ---------------------------------------------------------------------------

def _try_load_cuda_mamba():
    try:
        from mamba_ssm import Mamba
        logger.info("Using CUDA-optimised mamba-ssm")
        return Mamba
    except ImportError:
        logger.info("mamba-ssm not available; using pure-PyTorch Mamba fallback")
        return None


# ---------------------------------------------------------------------------
# Longitudinal Mamba model
# ---------------------------------------------------------------------------

class MambaLongitudinalModel(nn.Module):
    """Mamba-based sequential model for multi-visit X-ray analysis.

    Input (per visit):
      - DINOv2 CLS embedding  : d_dino dimensions
      - Detection features     : n_det_features dimensions (fracture confidence, bbox area, etc.)
      - Segmentation features  : n_seg_features dimensions (mask area, etc.)
      - Time delta             : 1 dimension (days since first study)
      - Clinical features      : n_clinical dimensions (age, gender, cast, etc.)

    Output (per visit):
      - healing_stage_logits : n_healing_stages class logits
      - sequence_repr        : global sequence representation (last hidden state)

    Parameters
    ----------
    d_dino          : DINOv2 embedding dimension (384/768/1024/1536)
    n_det_features  : number of detection feature scalars
    n_seg_features  : number of segmentation feature scalars
    n_clinical      : number of structured clinical feature scalars
    d_model         : Mamba internal model dimension
    d_state         : Mamba state dimension
    d_conv          : Mamba conv kernel size
    expand          : Mamba expansion factor
    n_layers        : number of Mamba blocks
    n_healing_stages: number of healing stage classes (output head)
    dropout         : dropout rate
    """

    def __init__(
        self,
        d_dino: int = 384,
        n_det_features: int = 5,
        n_seg_features: int = 3,
        n_clinical: int = 4,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        n_healing_stages: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_dino = d_dino
        self.n_healing_stages = n_healing_stages
        d_input = d_dino + n_det_features + n_seg_features + 1 + n_clinical  # +1 for time delta

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mamba blocks
        CudaMamba = _try_load_cuda_mamba()
        mamba_blocks = []
        for _ in range(n_layers):
            if CudaMamba is not None:
                try:
                    block = CudaMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                    mamba_blocks.append(block)
                    continue
                except Exception:
                    pass
            mamba_blocks.append(MambaBlockPyTorch(d_model, d_state, d_conv, expand))
        self.mamba_layers = nn.ModuleList(mamba_blocks)
        self.norm = nn.LayerNorm(d_model)

        # Output head: healing-stage classification
        self.healing_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_healing_stages),
        )

        logger.info(
            "MambaLongitudinalModel: d_input=%d → d_model=%d, %d layers, %d healing stages",
            d_input, d_model, n_layers, n_healing_stages,
        )

    def forward(
        self,
        embeddings: torch.Tensor,      # (B, L, d_dino)
        det_features: torch.Tensor,    # (B, L, n_det_features)
        seg_features: torch.Tensor,    # (B, L, n_seg_features)
        time_deltas: torch.Tensor,     # (B, L, 1)
        clinical_features: torch.Tensor,  # (B, L, n_clinical)
        padding_mask: torch.Tensor | None = None,  # (B, L) bool, True = padded
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict:
            "healing_logits"  : (B, L, n_healing_stages)
            "sequence_repr"   : (B, d_model) — last non-padded timestep
            "hidden_states"   : (B, L, d_model)
        """
        x = torch.cat([embeddings, det_features, seg_features, time_deltas, clinical_features], dim=-1)
        x = self.input_proj(x)       # (B, L, d_model)

        for mamba in self.mamba_layers:
            x = mamba(x)
        x = self.norm(x)             # (B, L, d_model)

        healing_logits = self.healing_head(x)  # (B, L, n_healing_stages)

        # Sequence representation = last non-padded hidden state
        if padding_mask is not None:
            seq_lengths = (~padding_mask).sum(dim=1) - 1  # last valid index
            seq_lengths = seq_lengths.clamp(min=0)
            seq_repr = x[torch.arange(x.shape[0]), seq_lengths]
        else:
            seq_repr = x[:, -1, :]  # last timestep

        return {
            "healing_logits": healing_logits,
            "sequence_repr": seq_repr,
            "hidden_states": x,
        }

    def predict(
        self,
        embeddings: torch.Tensor,
        det_features: torch.Tensor,
        seg_features: torch.Tensor,
        time_deltas: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> dict[str, Any]:
        """Inference with softmax probabilities."""
        self.eval()
        with torch.no_grad():
            out = self.forward(embeddings, det_features, seg_features, time_deltas, clinical_features)
        probs = torch.softmax(out["healing_logits"], dim=-1)
        return {
            "healing_stage_probabilities": probs,
            "sequence_representation": out["sequence_repr"],
            "clinically_validated": False,
            "recovery_training_enabled": False,
            "note": (
                "Healing-stage probabilities are model estimates based on "
                "fracture_visible label progression. They are NOT clinically "
                "validated and must not be used for clinical decisions."
            ),
        }
