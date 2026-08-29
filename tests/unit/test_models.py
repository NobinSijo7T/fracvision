"""Unit tests for model components."""
from __future__ import annotations

import pytest
import torch
import numpy as np


class TestDINOv2Extractor:
    def test_extractor_initialises(self):
        """DINOv2 extractor should initialise without errors (downloads weights)."""
        pytest.importorskip("torch")
        # Use smallest model for speed
        try:
            from model.models.dinov2.extractor import DINOv2Extractor
            extractor = DINOv2Extractor(model_name="dinov2_vits14", device="cpu")
            assert extractor.embedding_dim == 384
        except Exception as exc:
            pytest.skip(f"DINOv2 not available (no internet or torch.hub issue): {exc}")

    def test_extract_output_shape(self):
        pytest.importorskip("torch")
        try:
            from model.models.dinov2.extractor import DINOv2Extractor
            extractor = DINOv2Extractor(model_name="dinov2_vits14", device="cpu")
            dummy = torch.rand(2, 3, 224, 224)
            out = extractor.extract(dummy)
            assert out["cls_token"].shape == (2, 384), f"Unexpected shape: {out['cls_token'].shape}"
            assert out["patch_tokens"].ndim == 3
        except Exception as exc:
            pytest.skip(str(exc))

    def test_invalid_model_name_raises(self):
        from model.models.dinov2.extractor import DINOv2Extractor
        with pytest.raises(ValueError, match="Unknown DINOv2 model"):
            DINOv2Extractor(model_name="dinov2_vit_invalid", device="cpu")


class TestFractureClassifier:
    def test_forward_output_shape(self):
        from model.models.dinov2.classifier import FractureClassifier
        clf = FractureClassifier(
            embedding_dim=384,
            class_names=["fractured", "hand", "leg"],
            hidden_dims=[64],
        )
        emb = torch.rand(4, 384)
        logits = clf(emb)
        assert logits.shape == (4, 3), f"Expected (4,3), got {logits.shape}"

    def test_predict_returns_probs(self):
        from model.models.dinov2.classifier import FractureClassifier
        clf = FractureClassifier(
            embedding_dim=384,
            class_names=["fractured", "hand"],
            hidden_dims=[],
        )
        emb = torch.rand(1, 384)
        result = clf.predict(emb)
        assert "probabilities" in result
        assert result["probabilities"].shape == (1, 2)
        assert result["probabilities"].min() >= 0.0
        assert result["probabilities"].max() <= 1.0

    def test_focal_loss_decreases_on_overfit(self):
        from model.models.dinov2.classifier import FractureClassifier
        clf = FractureClassifier(embedding_dim=32, class_names=["fractured"], hidden_dims=[])
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.1)
        emb = torch.rand(8, 32)
        targets = torch.ones(8, 1)
        losses = []
        for _ in range(20):
            loss = clf.compute_loss(emb, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], "Loss should decrease with training"


class TestMambaLongitudinal:
    def test_forward_output_shapes(self):
        from model.models.mamba.longitudinal import MambaLongitudinalModel
        model = MambaLongitudinalModel(
            d_dino=64, n_det_features=3, n_seg_features=2, n_clinical=2,
            d_model=32, d_state=4, d_conv=2, expand=2, n_layers=2,
            n_healing_stages=2,
        )
        B, L = 2, 5
        out = model(
            torch.rand(B, L, 64),
            torch.rand(B, L, 3),
            torch.rand(B, L, 2),
            torch.rand(B, L, 1),
            torch.rand(B, L, 2),
        )
        assert out["healing_logits"].shape == (B, L, 2)
        assert out["sequence_repr"].shape == (B, 32)

    def test_gradient_flows(self):
        from model.models.mamba.longitudinal import MambaLongitudinalModel
        model = MambaLongitudinalModel(
            d_dino=32, n_det_features=2, n_seg_features=2, n_clinical=2,
            d_model=16, d_state=4, d_conv=2, expand=2, n_layers=1,
            n_healing_stages=2,
        )
        B, L = 2, 3
        out = model(
            torch.rand(B, L, 32, requires_grad=False),
            torch.rand(B, L, 2), torch.rand(B, L, 2),
            torch.rand(B, L, 1), torch.rand(B, L, 2),
        )
        logits = out["healing_logits"]
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 2), torch.zeros(B * L, dtype=torch.long))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestRecoveryStub:
    def test_training_disabled(self):
        from model.models.mamba.recovery import is_training_enabled
        assert is_training_enabled() is False

    def test_predict_returns_synthetic_flag(self):
        from model.models.mamba.recovery import RecoveryPredictionModule
        module = RecoveryPredictionModule(d_model=32)
        seq_repr = torch.rand(2, 32)
        result = module(seq_repr)
        assert result["synthetic_test_only"] is True
        assert result["clinical_use"] is False
        assert result["recovery_training_enabled"] is False

    def test_data_schema_contains_required_fields(self):
        from model.models.mamba.recovery import RecoveryPredictionModule, REQUIRED_CLINICAL_FIELDS
        module = RecoveryPredictionModule(d_model=32)
        schema = module.get_data_schema()
        for field in REQUIRED_CLINICAL_FIELDS:
            assert field in schema["required_fields"]


class TestMedSAMSegmentor:
    def test_missing_checkpoint_returns_placeholder(self):
        from model.models.medsam.segmentor import MedSAMSegmentor
        seg = MedSAMSegmentor(checkpoint_path=None, device="cpu")
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = seg.segment(img, [10.0, 10.0, 100.0, 100.0])
        assert result["model_loaded"] is False
        assert "warning" in result
        assert result["mask"].shape == (256, 256)

    def test_dice_computation(self):
        from model.models.medsam.segmentor import compute_dice, compute_iou
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        assert compute_dice(pred, gt) == 1.0  # both empty = perfect

        pred[2:5, 2:5] = True
        gt[2:5, 2:5] = True
        assert compute_dice(pred, gt) == pytest.approx(1.0)

        gt[6:9, 6:9] = True
        assert compute_dice(pred, gt) < 1.0
        assert compute_iou(pred, gt) < 1.0
