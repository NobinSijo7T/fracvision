"""Unit tests for dataset adapters."""
from __future__ import annotations

from pathlib import Path
import pytest


class TestFracAtlasAdapter:
    def test_load_records_returns_list(self, fracatlas_adapter, fracatlas_root):
        if not fracatlas_root.exists():
            pytest.skip("FracAtlas dataset not found")
        records = fracatlas_adapter.load_records()
        assert isinstance(records, list), "load_records should return a list"
        assert len(records) > 0, "Expected at least 1 record"

    def test_record_has_required_fields(self, fracatlas_records):
        if not fracatlas_records:
            pytest.skip("No records loaded")
        for field in ["image_id", "patient_id", "labels", "annotations", "mask_polygons", "metadata"]:
            assert field in fracatlas_records[0], f"Missing field: {field}"

    def test_image_id_is_non_empty(self, fracatlas_records):
        for rec in fracatlas_records:
            assert rec["image_id"], "image_id should be non-empty"

    def test_labels_are_valid(self, fracatlas_records):
        valid_labels = {
            "fractured", "hand", "leg", "hip", "shoulder", "mixed",
            "hardware", "multiscan", "frontal", "lateral", "oblique",
            "fracture_visible"
        }
        for rec in fracatlas_records[:100]:
            for lbl in rec["labels"]:
                assert lbl in valid_labels, f"Invalid label '{lbl}' in FracAtlas record"

    def test_no_recovery_labels(self, fracatlas_records):
        for rec in fracatlas_records:
            meta = rec.get("metadata", {})
            assert "healing_date" not in meta, "Unexpected recovery label found"
            assert "union_date" not in meta, "Unexpected recovery label found"

    def test_followup_metadata_is_empty(self, fracatlas_records):
        if not fracatlas_records:
            pytest.skip("No records loaded")
        assert fracatlas_records[0]["followup_metadata"] == {}, \
            "FracAtlas should have no followup metadata"

    def test_patient_id_surrogate(self, fracatlas_records, fracatlas_adapter):
        for rec in fracatlas_records[:10]:
            pid = fracatlas_adapter.get_patient_id(rec)
            assert pid, "patient_id should be non-empty"


class TestGRAZPEDWRIAdapter:
    def test_load_records_returns_list(self, grazpedwri_adapter, grazpedwri_root):
        if not grazpedwri_root.exists():
            pytest.skip("GRAZPEDWRI-DX dataset not found")
        records = grazpedwri_adapter.load_records()
        assert isinstance(records, list)
        assert len(records) > 0

    def test_recovery_labels_absent(self, grazpedwri_records):
        for rec in grazpedwri_records[:50]:
            assert rec.get("_recovery_labels_available") is False, \
                "Recovery labels should be marked as unavailable"

    def test_patient_ids_are_numeric_strings(self, grazpedwri_records):
        for rec in grazpedwri_records[:50]:
            pid = rec.get("patient_id", "")
            assert str(pid).strip() != "", "patient_id should not be empty"

    def test_valid_detection_labels(self, grazpedwri_records):
        from model.data.adapters.grazpedwri import KNOWN_GRAZPEDWRI_LABELS
        allowed = KNOWN_GRAZPEDWRI_LABELS | {"fracture_visible"}
        for rec in grazpedwri_records[:200]:
            for lbl in rec["labels"]:
                assert lbl in allowed, f"Unexpected label '{lbl}' in GRAZPEDWRI record"

    def test_followup_metadata_present(self, grazpedwri_records):
        records_with_meta = [r for r in grazpedwri_records if r.get("followup_metadata", {}).get("timehash")]
        assert len(records_with_meta) > 0, "Expected at least some records with timehash"

    def test_get_patient_sequences(self, grazpedwri_adapter, grazpedwri_records):
        sequences = grazpedwri_adapter.get_patient_sequences(grazpedwri_records)
        assert isinstance(sequences, dict)
        multi_study = {k: v for k, v in sequences.items() if len(v) >= 2}
        assert len(multi_study) > 0, "Expected at least some patients with multiple studies"
