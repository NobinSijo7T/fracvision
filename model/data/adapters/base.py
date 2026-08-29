from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetAdapter(ABC):
    """Small common interface; fields remain None where a dataset has no label."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @abstractmethod
    def load_records(self) -> list[dict[str, Any]]: ...

    def load_annotations(self) -> list[dict[str, Any]]:
        return [r for r in self.load_records() if r.get("annotations")]

    def get_image_path(self, record: dict[str, Any]) -> Path | None:
        value = record.get("image_path")
        return Path(value) if value else None

    def get_patient_id(self, record: dict[str, Any]) -> str:
        return str(record.get("patient_id") or record.get("image_id"))

    def get_study_id(self, record: dict[str, Any]) -> str | None:
        value = record.get("study_id")
        return str(value) if value is not None else None

    def get_labels(self, record: dict[str, Any]) -> list[str]:
        return list(record.get("labels") or [])

    def get_bbox(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        return list(record.get("annotations") or [])

    def get_mask(self, record: dict[str, Any]) -> Any:
        return record.get("mask")

    def get_followup_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record.get("followup_metadata") or {})
