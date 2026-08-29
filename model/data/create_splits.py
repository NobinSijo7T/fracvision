"""Deterministic patient-level train / validation / test split creator.

Rules enforced:
  - All images from the same patient are assigned to the same split.
  - For GRAZPEDWRI-DX, all studies from the same patient stay together.
  - Splits use the configured seed for full reproducibility.
  - Results are persisted to CSV so every experiment uses the same split.

Usage:
  python model/data/create_splits.py
  python model/data/create_splits.py --config model/configs/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from model.data.adapters import FracAtlasAdapter, GRAZPEDWRIAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def create_splits(
    records: list[dict],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[dict]]:
    """Assign each record to train/val/test using patient-level grouping.

    Parameters
    ----------
    records : list of record dicts with 'patient_id' key
    seed    : random seed for reproducibility
    train_frac : fraction of patients in training set
    val_frac   : fraction of patients in validation set (remainder → test)
    """
    # Group by patient_id
    patient_groups: dict[str, list[dict]] = {}
    for r in records:
        pid = str(r.get("patient_id", r.get("image_id", "?")))
        patient_groups.setdefault(pid, []).append(r)

    patient_ids = sorted(patient_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    logger.info(
        "Split: %d train patients / %d val patients / %d test patients "
        "(total %d patients, %d records)",
        len(train_ids), len(val_ids), len(test_ids), n, len(records),
    )

    def assign(pid: str) -> str:
        if pid in train_ids:
            return "train"
        if pid in val_ids:
            return "val"
        return "test"

    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for pid, pid_records in patient_groups.items():
        split = assign(pid)
        for r in pid_records:
            result[split].append({**r, "split": split})

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level dataset splits.")
    parser.add_argument("--config", default="model/configs/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = cfg.get("seed", 42)
    data_cfg = cfg.get("data", {})

    fracatlas_path = os.environ.get("FRACATLAS_ROOT", data_cfg.get("fracatlas_path", ""))
    grazpedwri_path = os.environ.get("GRAZPEDWRI_ROOT", data_cfg.get("grazpedwri_path", ""))

    adapters = [
        FracAtlasAdapter(fracatlas_path),
        GRAZPEDWRIAdapter(grazpedwri_path),
    ]

    all_records: list[dict] = []
    for adapter in adapters:
        dataset_name = adapter.__class__.__name__
        try:
            records = adapter.load_records()
        except Exception as exc:
            logger.error("Failed to load %s: %s", dataset_name, exc)
            continue
        for r in records:
            all_records.append({
                "dataset": dataset_name,
                "image_id": r.get("image_id", ""),
                "patient_id": adapter.get_patient_id(r),
                "study_id": adapter.get_study_id(r) or "",
                "image_path": r.get("image_path") or "",
                "fractured": "1" if "fractured" in (r.get("labels") or []) or "fracture_visible" in (r.get("labels") or []) else "0",
            })

    if not all_records:
        logger.error("No records loaded — check dataset paths in config.")
        sys.exit(1)

    splits = create_splits(all_records, seed=seed)

    out_dir = Path(data_cfg.get("split_dir", "model/data/splits"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "image_id", "patient_id", "study_id", "image_path", "fractured", "split"]
    for split_name, split_records in splits.items():
        out_path = out_dir / f"{split_name}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_records)
        logger.info("Wrote %d records to %s", len(split_records), out_path)

    print(f"\n✓ Patient-level splits written to {out_dir}/")
    for split_name, split_records in splits.items():
        print(f"  {split_name:6s}: {len(split_records):>6} records")


if __name__ == "__main__":
    main()
