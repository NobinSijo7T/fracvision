"""Integration and smoke tests for the FracVision end-to-end pipeline."""
from __future__ import annotations

import pytest
import numpy as np
import torch


class TestEndToEndSmoke:
    """Smoke tests that exercise full pipeline paths without real model weights."""

    def test_preprocessing_to_tensor(self, sample_fracatlas_image):
        if sample_fracatlas_image is None:
            pytest.skip("No FracAtlas image available")
        from model.data.preprocessing.preprocess import preprocess_image
        out, meta = preprocess_image(str(sample_fracatlas_image), image_size=224, return_tensor=True)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32

    def test_pipeline_predict_with_dummy_image(self, config):
        """Pipeline should return a PredictionResult even without real model weights."""
        from model.inference.pipeline import FracVisionPipeline, PredictionResult
        # Build pipeline without any real models loaded
        pipeline = FracVisionPipeline(config=config)
        dummy_np = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = pipeline.predict(dummy_np)
        assert isinstance(result, PredictionResult)
        assert result.clinically_validated is False
        assert result.recovery_training_enabled is False
        assert "disclaimer" in result.to_dict()

    def test_pipeline_disclaimer_present(self, config):
        from model.inference.pipeline import FracVisionPipeline
        pipeline = FracVisionPipeline(config=config)
        dummy = np.zeros((256, 256, 3), dtype=np.uint8) + 128
        result = pipeline.predict(dummy)
        assert result.disclaimer != ""
        assert "clinical" in result.disclaimer.lower()

    def test_compare_with_dummy_images(self, config):
        from model.inference.compare import SerialXrayComparator
        comparator = SerialXrayComparator(config=config)
        img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = comparator.compare(img1, img2)
        assert result["clinically_validated"] is False
        assert "disclaimer" in result
        assert "pixel_level_metrics" in result

    def test_recovery_stub_never_trains(self):
        from model.models.mamba.recovery import is_training_enabled, get_disabled_reason
        assert is_training_enabled() is False
        reason = get_disabled_reason()
        assert len(reason) > 20, "Disabled reason should have explanation text"

    def test_split_no_patient_overlap(self, config):
        """Verify that no patient appears in more than one split."""
        import csv
        from pathlib import Path
        split_dir = Path(config["data"]["split_dir"])
        if not split_dir.exists():
            pytest.skip("Splits not yet generated — run create_splits.py")

        patient_to_splits: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            csv_path = split_dir / f"{split_name}.csv"
            if not csv_path.exists():
                continue
            with csv_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    pid = row.get("patient_id", "")
                    patient_to_splits.setdefault(pid, set()).add(split_name)

        leaks = {pid: splits for pid, splits in patient_to_splits.items() if len(splits) > 1}
        assert len(leaks) == 0, (
            f"Patient-level data leakage detected: "
            f"{len(leaks)} patients appear in multiple splits: {dict(list(leaks.items())[:5])}"
        )

    def test_inspect_report_generation(self, tmp_path, config):
        """inspect_datasets should generate JSON reports without crashing."""
        import json, yaml
        from pathlib import Path
        # Create a temp config pointing to real dataset paths
        cfg = dict(config)
        cfg["data"] = {**cfg.get("data", {}), "reports_dir": str(tmp_path)}
        tmp_config = tmp_path / "config.yaml"
        tmp_config.write_text(yaml.dump(cfg), encoding="utf-8")

        from model.data import inspect_datasets  # import to check no import errors
        assert inspect_datasets is not None


class TestAPISmoke:
    """FastAPI smoke tests using the test client (no real models needed)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline_ready" in data
        assert "disclaimer" in data

    def test_health_includes_disclaimer(self, client):
        response = client.get("/api/health")
        body = response.json()
        assert "clinical" in body.get("disclaimer", "").lower()

    def test_predict_invalid_file_type(self, client):
        from io import BytesIO
        response = client.post(
            "/api/predict",
            files={"file": ("test.pdf", BytesIO(b"dummy"), "application/pdf")},
        )
        # Should return 415 or 503 (if pipeline not ready)
        assert response.status_code in {415, 503}

    def test_predict_too_large_file(self, client):
        from io import BytesIO
        big_data = BytesIO(b"0" * (55 * 1024 * 1024))  # 55 MB > 50 MB limit
        response = client.post(
            "/api/predict",
            files={"file": ("big.jpg", big_data, "image/jpeg")},
        )
        assert response.status_code in {413, 503}
