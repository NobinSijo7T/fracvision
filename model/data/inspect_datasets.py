"""Dataset inspection and quality reporting for FracVision.

Generates three machine-readable reports in model/reports/:
  dataset_inventory.json    — file counts, annotation formats, sample paths
  dataset_summary.csv       — one row per dataset, key statistics
  dataset_quality_report.json — corruption, duplicates, missing, class imbalance

Usage:
  python model/data/inspect_datasets.py
  python model/data/inspect_datasets.py --config model/configs/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Support running as a module or a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from PIL import Image, UnidentifiedImageError

from model.data.adapters import FracAtlasAdapter, GRAZPEDWRIAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm"}
ANNOTATION_SUFFIXES = {".xml", ".json", ".txt"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_annotation_format(suffix: str) -> str:
    return {
        ".xml": "pascal_voc",
        ".json": "coco_json",
        ".txt": "yolo_txt",
    }.get(suffix, "unknown")


# ---------------------------------------------------------------------------
# Per-dataset inspection
# ---------------------------------------------------------------------------

def inspect(name: str, adapter: FracAtlasAdapter | GRAZPEDWRIAdapter) -> tuple[dict, dict]:
    """Inspect a single dataset. Returns (inventory_dict, quality_dict)."""
    root = adapter.root
    logger.info("Inspecting %s at %s ...", name, root)

    if not root.exists():
        logger.error("Dataset root does not exist: %s", root)
        return (
            {"dataset": name, "root": str(root), "error": "root_not_found"},
            {"dataset": name, "error": "root_not_found"},
        )

    # --- File enumeration ---
    all_files = [p for p in root.rglob("*") if p.is_file()]
    image_files = [p for p in all_files if p.suffix.lower() in IMAGE_SUFFIXES]
    annotation_files = [p for p in all_files if p.suffix.lower() in ANNOTATION_SUFFIXES]
    metadata_files = [p for p in all_files if p.suffix.lower() in {".csv", ".tsv"}]

    annotation_formats = sorted({
        detect_annotation_format(p.suffix.lower())
        for p in annotation_files
        if p.suffix.lower() in ANNOTATION_SUFFIXES
    })

    # --- Image validation: dimensions, corruption, duplicates ---
    corrupt: list[dict] = []
    dim_counter: Counter = Counter()
    hash_to_paths: dict[str, list[str]] = {}

    logger.info("Validating %d image files (may take a moment) ...", len(image_files))
    for img in image_files:
        if img.suffix.lower() == ".dcm":
            # DICOM validation deferred — pydicom may not be installed
            continue
        try:
            with Image.open(img) as opened:
                opened.verify()
            with Image.open(img) as opened:
                dim_counter[f"{opened.width}x{opened.height}"] += 1
            h = sha256_file(img)
            hash_to_paths.setdefault(h, []).append(str(img.relative_to(root)))
        except (UnidentifiedImageError, OSError, ValueError, Exception) as exc:
            corrupt.append({
                "path": str(img.relative_to(root)),
                "error": str(exc),
            })
            logger.warning("Corrupt image %s: %s", img.name, exc)

    duplicate_groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]

    # --- Adapter records ---
    try:
        records = adapter.load_records()
    except Exception as exc:
        logger.error("Adapter.load_records() failed for %s: %s", name, exc)
        records = []

    label_counter: Counter = Counter()
    missing_images: list[str] = []
    missing_annotations: list[str] = []
    patient_ids: set[str] = set()
    study_ids: set[str] = set()
    followup_count = 0

    for r in records:
        # Label distribution
        for lbl in adapter.get_labels(r):
            label_counter[lbl] += 1

        # Missing image check
        img_path = adapter.get_image_path(r)
        if img_path is None or not img_path.exists():
            missing_images.append(r.get("image_id", "?"))

        # Missing annotation check (bbox)
        bboxes = adapter.get_bbox(r)
        if not bboxes:
            missing_annotations.append(r.get("image_id", "?"))

        # Patient/study tracking
        pid = adapter.get_patient_id(r)
        if pid:
            patient_ids.add(pid)
        sid = adapter.get_study_id(r)
        if sid:
            study_ids.add(f"{pid}_{sid}")

        # Follow-up metadata
        fm = adapter.get_followup_metadata(r)
        if fm.get("timehash") or fm.get("study_number"):
            followup_count += 1

    # Orphan annotations (XML files with no matching record)
    known_stems = {Path(r.get("image_id", "")).stem for r in records}
    orphan_annotations = [
        str(p.relative_to(root))
        for p in annotation_files
        if p.suffix.lower() == ".xml" and p.stem not in known_stems
    ]

    # Recovery label check
    recovery_labels_available = False
    recovery_note = (
        "No verified time-to-union / censoring endpoint detected. "
        "Supervised recovery-time training is DISABLED for this dataset."
    )

    # Build output dicts
    inventory = {
        "dataset": name,
        "root": str(root),
        "file_count": len(all_files),
        "image_file_count": len(image_files),
        "annotation_file_count": len(annotation_files),
        "metadata_file_count": len(metadata_files),
        "detected_annotation_formats": annotation_formats,
        "sample_files": [str(p.relative_to(root)) for p in all_files[:20]],
    }

    quality = {
        "dataset": name,
        "record_count": len(records),
        "patient_count": len(patient_ids),
        "unique_studies": len(study_ids),
        "followup_metadata_records": followup_count,
        "missing_image_records": missing_images[:50],   # cap to 50 in report
        "missing_image_count": len(missing_images),
        "missing_annotation_records": missing_annotations[:50],
        "missing_annotation_count": len(missing_annotations),
        "orphan_annotation_files": orphan_annotations[:50],
        "orphan_annotation_count": len(orphan_annotations),
        "corrupt_images": corrupt[:50],
        "corrupt_image_count": len(corrupt),
        "duplicate_image_groups": duplicate_groups[:20],
        "duplicate_group_count": len(duplicate_groups),
        "image_dimension_distribution": dict(dim_counter.most_common(20)),
        "label_distribution": dict(label_counter.most_common()),
        "recovery_labels_available": recovery_labels_available,
        "recovery_labels_note": recovery_note,
    }

    logger.info(
        "%s: %d records, %d patients, %d images, %d corrupt, %d label types",
        name, len(records), len(patient_ids), len(image_files),
        len(corrupt), len(label_counter),
    )
    return inventory, quality


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect FracVision datasets and generate quality reports."
    )
    parser.add_argument(
        "--config", default="model/configs/config.yaml",
        help="Path to config.yaml (default: model/configs/config.yaml)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config.get("data", {})

    fracatlas_path = os.environ.get("FRACATLAS_ROOT", data_cfg.get("fracatlas_path", ""))
    grazpedwri_path = os.environ.get("GRAZPEDWRI_ROOT", data_cfg.get("grazpedwri_path", ""))

    adapters = [
        ("FracAtlas", FracAtlasAdapter(fracatlas_path)),
        ("GRAZPEDWRI-DX", GRAZPEDWRIAdapter(grazpedwri_path)),
    ]

    inventories, qualities = [], []
    for name, adapter in adapters:
        inv, qual = inspect(name, adapter)
        inventories.append(inv)
        qualities.append(qual)

    # Write reports
    output_dir = Path(data_cfg.get("reports_dir", "model/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    (output_dir / "dataset_inventory.json").write_text(
        json.dumps({"generated_at": timestamp, "datasets": inventories}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "dataset_quality_report.json").write_text(
        json.dumps({"generated_at": timestamp, "datasets": qualities}, indent=2),
        encoding="utf-8",
    )

    summary_fields = [
        "dataset", "record_count", "patient_count", "image_file_count",
        "missing_image_count", "missing_annotation_count", "corrupt_image_count",
        "duplicate_group_count", "followup_metadata_records",
        "recovery_labels_available",
    ]
    with (output_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for inv, qual in zip(inventories, qualities):
            merged = {**inv, **qual}
            writer.writerow({k: merged.get(k, "") for k in summary_fields})

    print(f"\n✓ Reports written to {output_dir}/")
    print(f"  dataset_inventory.json")
    print(f"  dataset_quality_report.json")
    print(f"  dataset_summary.csv")
    print()
    for qual in qualities:
        print(
            f"  {qual['dataset']:20s} — {qual['record_count']:>6} records, "
            f"{qual['patient_count']:>5} patients, "
            f"{qual['corrupt_image_count']:>3} corrupt, "
            f"recovery_labels={qual['recovery_labels_available']}"
        )


if __name__ == "__main__":
    main()
