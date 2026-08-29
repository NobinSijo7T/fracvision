"""Recovery prediction module — INFERENCE-READY STUB.

⚠️  SUPERVISED TRAINING IS DISABLED ⚠️

Neither FracAtlas nor GRAZPEDWRI-DX provides verified time-to-union or
censoring-status labels. This module exists as a well-defined interface
that can be activated when valid longitudinal clinical outcome data becomes
available.

All outputs from this module are clearly labeled:
  "synthetic_test_only": true
  "clinical_use": false
  "recovery_training_enabled": false

Required schema for future supervised training
-----------------------------------------------
When clinical labels become available, the following fields are required
per imaging study record:
  patient_id             : str  — de-identified patient identifier
  injury_id              : str  — unique injury identifier
  injury_date            : date — date of injury / initial imaging
  xray_date              : date — date of this X-ray
  follow_up_status       : str  — "active" | "healed" | "censored"
  healing_or_union_date  : date | None — date of confirmed radiographic union
  censoring_date         : date | None — date of last known follow-up (if censored)
  ao_classification      : str | None — AO fracture classification code

Until these fields are provided, call is_training_enabled() to confirm
disabled status before any training attempt.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Required fields for supervised recovery training
REQUIRED_CLINICAL_FIELDS = [
    "patient_id",
    "injury_id",
    "injury_date",
    "xray_date",
    "follow_up_status",
    "healing_or_union_date",
    "censoring_date",
]

_DISABLED_REASON = (
    "Supervised recovery-time training is DISABLED. "
    "Neither FracAtlas nor GRAZPEDWRI-DX contains verified time-to-union "
    "or censoring-status labels. Provide clinical outcome data matching "
    "the schema in REQUIRED_CLINICAL_FIELDS to enable this module."
)


def is_training_enabled() -> bool:
    """Always returns False until valid clinical outcome labels are provided."""
    return False


def get_disabled_reason() -> str:
    return _DISABLED_REASON


class RecoveryPredictionModule(nn.Module):
    """Time-to-event (survival) prediction head — STUB IMPLEMENTATION.

    Interface is fully defined and code-complete. Supervised training
    is gated behind is_training_enabled() which returns False.

    When activated (future work):
      - Input: Mamba sequence representation (B, d_model)
      - Output: estimated time-to-radiographic-union + uncertainty
      - Loss: survival / time-to-event loss (censoring-aware)
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self._training_enabled = False  # hard-coded False

        # Prediction head (architecture is ready; weights are not trained)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # scalar time estimate
        )
        self.log_var = nn.Linear(d_model, 1)  # uncertainty head

        logger.warning(
            "RecoveryPredictionModule initialised in STUB mode. "
            "Training is disabled — %s", _DISABLED_REASON
        )

    def forward(self, sequence_repr: torch.Tensor) -> dict[str, Any]:
        """Return CLEARLY LABELLED synthetic test outputs.

        Parameters
        ----------
        sequence_repr : (B, d_model) from MambaLongitudinalModel

        Returns
        -------
        dict — all values are clearly flagged as non-clinical test outputs
        """
        if self._training_enabled:
            raise RuntimeError(
                "Training is not supported until clinical labels are provided. "
                "This guard should never be reached."
            )

        # Generate structurally-correct but meaningless outputs for smoke tests
        B = sequence_repr.shape[0]
        dummy_time = torch.full((B, 1), fill_value=-1.0)  # -1 flags as invalid
        dummy_uncertainty = torch.full((B, 1), fill_value=float("nan"))

        return {
            "estimated_days_to_union": dummy_time.tolist(),
            "uncertainty": dummy_uncertainty.tolist(),
            "prediction_interval_lower": None,
            "prediction_interval_upper": None,
            "synthetic_test_only": True,
            "clinical_use": False,
            "recovery_training_enabled": False,
            "disabled_reason": _DISABLED_REASON,
            "required_fields_for_activation": REQUIRED_CLINICAL_FIELDS,
        }

    def get_data_schema(self) -> dict[str, Any]:
        """Return the required data schema for future supervised training."""
        return {
            "required_fields": REQUIRED_CLINICAL_FIELDS,
            "field_descriptions": {
                "patient_id": "De-identified patient identifier (string)",
                "injury_id": "Unique identifier per injury episode",
                "injury_date": "ISO 8601 date of injury or initial presentation",
                "xray_date": "ISO 8601 date of this imaging study",
                "follow_up_status": "One of: 'active', 'healed', 'censored'",
                "healing_or_union_date": "ISO 8601 date of confirmed radiographic union (null if not healed/censored)",
                "censoring_date": "ISO 8601 date of last known follow-up (null if healed)",
            },
            "notes": [
                "All dates must be verified from clinical records — do not fabricate.",
                "Censored observations (lost to follow-up) must be correctly marked.",
                "AO classification codes are optional but improve model accuracy.",
                "This schema must be reviewed by a clinical collaborator before use.",
            ],
        }
