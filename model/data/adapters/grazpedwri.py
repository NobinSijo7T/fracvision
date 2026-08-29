"""GRAZPEDWRI-DX dataset adapter.

Confirmed dataset structure (inspected 2026-08-29):
  dataset.csv columns:
    filestem, patient_id, study_number, timehash, gender, age,
    laterality, projection, initial_exam, ao_classification, cast,
    diagnosis_uncertain, osteopenia, fracture_visible, metal,
    pixel_spacing, device_manufacturer
  Images: .png files inside folder_structure.zip
    (auto-extracted to folder_structure/images/ by this adapter)
  Annotations:
    folder_structure/pascalvoc/*.xml — Pascal VOC format
    Labels confirmed from inspection:
      fracture, text, pronatorsign, periostealreaction,
      softissue, boneoutline, hardwareother

Patient IDs:
  Numeric patient_id column — real patient-level splitting available.
  Multiple study_number values per patient → true longitudinal data.

Recovery labels: ABSENT.
  timehash timestamps exist, but no time-to-union / censoring endpoint.
  Supervised recovery-time training is DISABLED.
"""
from __future__ import annotations

import csv
import logging
import zipfile
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from .xml import parse_voc

logger = logging.getLogger(__name__)

# Labels confirmed from GRAZPEDWRI-DX VOC annotation inspection
KNOWN_GRAZPEDWRI_LABELS = frozenset({
    "fracture",
    "pronatorsign",
    "periostealreaction",
    "text",
    "softissue",
    "boneoutline",
    "hardwareother",
})


class GRAZPEDWRIAdapter(DatasetAdapter):
    """Adapter for the GRAZPEDWRI-DX wrist X-ray dataset.

    Auto-extracts images from folder_structure.zip if the images directory
    does not already exist. Never overwrites original data files.
    """

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self._images_dir: Path | None = None  # resolved lazily

    # ------------------------------------------------------------------
    # Image discovery (with auto-extraction from zip)
    # ------------------------------------------------------------------

    def _ensure_images_extracted(self) -> Path:
        """Return the directory containing extracted .png images.

        Tries (in order):
        1. folder_structure/images/  (already extracted)
        2. Extract folder_structure.zip → folder_structure/images/
        Falls back to a broad rglob scan if neither applies.
        """
        images_dir = self.root / "folder_structure" / "images"
        if images_dir.exists() and any(images_dir.rglob("*.png")):
            return images_dir

        zip_path = self.root / "folder_structure.zip"
        if zip_path.exists() and not images_dir.exists():
            logger.info(
                "Extracting %s to %s (this may take a while) ...",
                zip_path.name,
                images_dir,
            )
            images_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    png_members = [m for m in zf.namelist() if m.lower().endswith(".png")]
                    for member in png_members:
                        target = images_dir / Path(member).name
                        if not target.exists():
                            data = zf.read(member)
                            target.write_bytes(data)
                logger.info("Extracted %d PNG files to %s", len(png_members), images_dir)
            except Exception as exc:
                logger.error("Failed to extract zip %s: %s", zip_path, exc)

        if images_dir.exists():
            return images_dir

        # Broad fallback: scan entire dataset root
        logger.warning(
            "GRAZPEDWRI images directory not found at %s and zip extraction failed. "
            "Falling back to recursive scan — this may be slow.",
            images_dir,
        )
        return self.root

    def _build_image_lookup(self) -> dict[str, Path]:
        """Build a stem → Path lookup for all .png image files."""
        images_dir = self._ensure_images_extracted()
        lookup: dict[str, Path] = {}
        for p in images_dir.rglob("*"):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"}:
                lookup[p.stem] = p
        logger.info("GRAZPEDWRI: found %d image files", len(lookup))
        return lookup

    # ------------------------------------------------------------------
    # Annotation discovery
    # ------------------------------------------------------------------

    def _build_annotation_map(self) -> dict[str, list[dict]]:
        """Build stem → list[bbox_dict] from Pascal VOC XMLs."""
        voc_dir = self.root / "folder_structure" / "pascalvoc"
        annotation_map: dict[str, list[dict]] = {}
        if not voc_dir.exists():
            logger.warning("GRAZPEDWRI VOC annotation dir not found: %s", voc_dir)
            return annotation_map

        xml_files = list(voc_dir.glob("*.xml"))
        logger.info("GRAZPEDWRI: parsing %d VOC XML files ...", len(xml_files))
        unknown_labels: set[str] = set()
        for xml_path in xml_files:
            try:
                filename, boxes = parse_voc(xml_path)
                stem = Path(filename or xml_path.stem).stem
                for box in boxes:
                    lbl = box.get("label", "")
                    if lbl and lbl not in KNOWN_GRAZPEDWRI_LABELS:
                        unknown_labels.add(lbl)
                    box["source"] = "pascal_voc"
                annotation_map.setdefault(stem, []).extend(boxes)
            except Exception as exc:
                logger.warning("Failed to parse VOC XML %s: %s", xml_path.name, exc)

        if unknown_labels:
            logger.info(
                "GRAZPEDWRI: found additional label(s) not in known set: %s",
                sorted(unknown_labels),
            )

        return annotation_map

    # ------------------------------------------------------------------
    # Public adapter interface
    # ------------------------------------------------------------------

    def load_records(self) -> list[dict[str, Any]]:
        """Load all GRAZPEDWRI-DX records.

        Each record contains:
          image_id, patient_id, study_id, image_path, labels,
          annotations (list of bbox dicts), metadata, followup_metadata.

        followup_metadata includes: timehash, initial_exam, projection,
          study_number, ao_classification — enabling longitudinal analysis.
        """
        # Load CSV
        csv_path = self.root / "dataset.csv"
        rows: list[dict[str, str]] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        else:
            logger.warning("GRAZPEDWRI dataset.csv not found at %s", csv_path)

        # Build image + annotation lookups (lazy)
        image_lookup = self._build_image_lookup()
        annotation_map = self._build_annotation_map()

        records = []
        missing_images = 0
        for row in rows:
            stem = row.get("filestem", "").strip()
            if not stem:
                continue

            img_path = image_lookup.get(stem)
            if img_path is None:
                missing_images += 1

            # Derive labels: fracture_visible is the primary binary label
            labels: list[str] = []
            if row.get("fracture_visible", "").strip() == "1":
                labels.append("fracture_visible")

            boxes = list(annotation_map.get(stem, []))
            # Also capture label names from annotation boxes as additional labels
            ann_labels = list({b["label"] for b in boxes if b.get("label")})
            for lbl in ann_labels:
                if lbl not in labels:
                    labels.append(lbl)

            records.append({
                "image_id": stem,
                "patient_id": row.get("patient_id", stem),
                "study_id": row.get("study_number"),
                "image_path": str(img_path) if img_path else None,
                "labels": labels,
                "annotations": boxes,
                "mask_polygons": [],  # GRAZPEDWRI has no polygon masks
                "metadata": row,
                "followup_metadata": {
                    "timehash": row.get("timehash"),
                    "study_number": row.get("study_number"),
                    "initial_exam": row.get("initial_exam"),
                    "projection": row.get("projection"),
                    "laterality": row.get("laterality"),
                    "ao_classification": row.get("ao_classification"),
                    "cast": row.get("cast"),
                    "age": row.get("age"),
                    "gender": row.get("gender"),
                },
                # Explicit confirmation: no recovery labels
                "_recovery_labels_available": False,
            })

        if missing_images:
            logger.warning(
                "GRAZPEDWRI: %d records have no resolved image file. "
                "Run zip extraction or verify folder_structure/images/ exists.",
                missing_images,
            )

        logger.info("GRAZPEDWRIAdapter: loaded %d records", len(records))
        return records

    def get_patient_sequences(self, records: list[dict[str, Any]]) -> dict[str, list[dict]]:
        """Return per-patient ordered list of records (by timehash).

        Used by the Mamba longitudinal model to build temporal sequences.
        All records must come from the same split to avoid data leakage.
        """
        from collections import defaultdict

        patient_map: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            pid = self.get_patient_id(record)
            patient_map[pid].append(record)

        # Sort each patient's records by timehash (ascending chronological order)
        def timehash_key(r: dict) -> int:
            th = r.get("followup_metadata", {}).get("timehash", "0")
            try:
                return int(th)
            except (ValueError, TypeError):
                return 0

        return {
            pid: sorted(records_list, key=timehash_key)
            for pid, records_list in patient_map.items()
        }
