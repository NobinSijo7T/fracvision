"""FracAtlas dataset adapter.

Confirmed dataset structure (inspected 2026-08-29):
  dataset.csv columns:
    image_id, hand, leg, hip, shoulder, mixed, hardware,
    multiscan, fractured, fracture_count, frontal, lateral, oblique
  Images:
    images/Fractured/     — JPG, fractured images
    images/Non_fractured/ — JPG, non-fractured images
  Annotations:
    Annotations/COCO JSON/COCO_fracture_masks.json  — polygon masks + bboxes
    Annotations/YOLO/IMG*.txt                       — YOLO format (class cx cy w h)
    Annotations/PASCAL VOC/*.xml                    — Pascal VOC bounding boxes

Patient IDs:
  FracAtlas has no separate patient ID column. image_id is used as surrogate.
  All images are therefore treated as independent (no longitudinal splits needed
  for this dataset).

Recovery labels: ABSENT. Supervised recovery training is disabled.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from .base import DatasetAdapter
from .coco_json import parse_coco_json
from .xml import parse_voc

logger = logging.getLogger(__name__)

# YOLO class 0 = fracture (confirmed from single-class YOLO files)
YOLO_CLASS_MAP = {0: "fracture"}

# Body-part columns in dataset.csv that represent one-hot labels
BODY_PART_COLS = ["hand", "leg", "hip", "shoulder", "mixed"]
VIEW_COLS = ["frontal", "lateral", "oblique"]
BINARY_COLS = ["fractured", "hardware", "multiscan"]
COUNT_COLS = ["fracture_count"]


class FracAtlasAdapter(DatasetAdapter):
    """Adapter for the FracAtlas dataset.

    Supports COCO JSON masks, YOLO TXT and Pascal VOC bounding boxes.
    Class names are derived exclusively from the inspected annotation files;
    no labels are invented.
    """

    def load_records(self) -> list[dict[str, Any]]:
        """Load all FracAtlas records with annotations and metadata.

        Returns a list of dicts, each representing one image with:
          image_id, patient_id, study_id, image_path, labels,
          annotations (list of bbox dicts), mask_polygons, metadata,
          followup_metadata (always empty for FracAtlas)
        """
        # 1. Load CSV metadata
        csv_path = self.root / "dataset.csv"
        rows: dict[str, dict[str, str]] = {}
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    image_id = row.get("image_id", "").strip()
                    if image_id:
                        rows[image_id] = row
        else:
            logger.warning("FracAtlas dataset.csv not found at %s", csv_path)

        # 2. Load COCO JSON masks
        coco_path = self.root / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"
        coco_data: dict[str, list[dict]] = {}
        if coco_path.exists():
            try:
                parsed = parse_coco_json(coco_path)
                coco_data = parsed["by_image"]
            except Exception as exc:
                logger.error("Failed to parse COCO JSON at %s: %s", coco_path, exc)
        else:
            logger.warning("FracAtlas COCO JSON not found at %s", coco_path)

        # 3. Load YOLO TXT annotations
        yolo_dir = self.root / "Annotations" / "YOLO"
        yolo_map: dict[str, list[dict]] = {}
        if yolo_dir.exists():
            for txt_path in yolo_dir.glob("*.txt"):
                stem = txt_path.stem
                boxes = []
                try:
                    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_idx, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            logger.debug("Invalid YOLO bbox in %s: %s", txt_path.name, parts)
                            continue
                        boxes.append({
                            "label": YOLO_CLASS_MAP.get(cls_idx, f"class_{cls_idx}"),
                            "bbox_yolo": [cx, cy, w, h],
                            "source": "yolo",
                        })
                except Exception as exc:
                    logger.warning("Failed to parse YOLO file %s: %s", txt_path, exc)
                yolo_map[stem] = boxes

        # 4. Load Pascal VOC annotations
        voc_dir = self.root / "Annotations" / "PASCAL VOC"
        voc_map: dict[str, list[dict]] = {}
        if voc_dir.exists():
            for xml_path in voc_dir.glob("*.xml"):
                try:
                    filename, boxes = parse_voc(xml_path)
                    stem = Path(filename or xml_path.stem).stem
                    for box in boxes:
                        box["source"] = "pascal_voc"
                    voc_map.setdefault(stem, []).extend(boxes)
                except Exception as exc:
                    logger.warning("Failed to parse VOC XML %s: %s", xml_path, exc)

        # 5. Build image path lookup (search both Fractured and Non_fractured dirs)
        image_lookup: dict[str, Path] = {}
        for subdir in ("Fractured", "Non_fractured"):
            img_dir = self.root / "images" / subdir
            if img_dir.exists():
                for img_path in img_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".dcm"}:
                        image_lookup[img_path.stem] = img_path
                        image_lookup[img_path.name] = img_path

        # 6. Merge all sources into records
        all_keys = set(rows) | set(yolo_map) | set(voc_map)
        # Also pick up images without annotations
        for img_id, img_path in image_lookup.items():
            all_keys.add(img_id)
            all_keys.add(img_path.name)

        records = []
        seen: set[str] = set()
        for raw_id in sorted(all_keys):
            # Normalise to image_id format (prefer row key with extension)
            if raw_id in rows:
                image_id = raw_id
            elif raw_id + ".jpg" in rows:
                image_id = raw_id + ".jpg"
            elif raw_id.rstrip(".jpg") in rows:
                image_id = raw_id
            else:
                image_id = raw_id

            if image_id in seen:
                continue
            seen.add(image_id)

            stem = Path(image_id).stem
            row = rows.get(image_id, rows.get(stem, {}))

            # Determine labels from CSV
            labels: list[str] = []
            for col in BODY_PART_COLS + VIEW_COLS + BINARY_COLS:
                if row.get(col, "0").strip() == "1":
                    labels.append(col)

            # Resolve image path
            img_path_resolved: Path | None = (
                image_lookup.get(image_id)
                or image_lookup.get(stem)
            )

            # Combine bboxes: YOLO (normalised) + VOC (absolute)
            bboxes = list(yolo_map.get(stem, [])) + list(voc_map.get(stem, []))

            # COCO mask polygons
            coco_anns = coco_data.get(image_id, coco_data.get(stem, []))

            records.append({
                "image_id": image_id,
                "patient_id": stem,          # surrogate — no real patient ID
                "study_id": None,
                "image_path": str(img_path_resolved) if img_path_resolved else None,
                "labels": labels,
                "annotations": bboxes,        # YOLO + VOC bboxes
                "mask_polygons": coco_anns,   # COCO segmentation masks
                "metadata": row,
                "followup_metadata": {},      # FracAtlas has no longitudinal data
            })

        logger.info("FracAtlasAdapter: loaded %d records", len(records))
        return records
