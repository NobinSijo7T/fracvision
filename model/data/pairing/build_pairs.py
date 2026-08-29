"""Build serial X-ray pair and sequence tables for GRAZPEDWRI-DX.

GRAZPEDWRI-DX contains multiple studies per patient (indexed by study_number
and timestamped by timehash). This script constructs:
  1. Per-patient ordered sequences (for Mamba longitudinal model)
  2. Consecutive-study pairs (for serial X-ray comparison module)

IMPORTANT: Pairs/sequences are built AFTER splitting. This script reads the
split CSVs to ensure no cross-split pairs are created (preventing data leakage).

Usage:
  python model/data/pairing/build_pairs.py
  python model/data/pairing/build_pairs.py --config model/configs/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml
from model.data.adapters import GRAZPEDWRIAdapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def build_sequences(
    records: list[dict],
    min_studies: int = 2,
) -> dict[str, list[dict]]:
    """Group records by patient and sort chronologically by timehash.

    Parameters
    ----------
    records     : list of GRAZPEDWRI-DX record dicts
    min_studies : only include patients with at least this many studies

    Returns
    -------
    dict: patient_id → sorted list of records
    """
    patient_map: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        pid = str(r.get("patient_id", ""))
        if pid:
            patient_map[pid].append(r)

    sequences = {}
    for pid, studies in patient_map.items():
        if len(studies) < min_studies:
            continue
        # Sort by timehash (ascending = chronological)
        sorted_studies = sorted(
            studies,
            key=lambda r: int(r.get("followup_metadata", {}).get("timehash") or r.get("timehash") or 0),
        )
        sequences[pid] = sorted_studies

    logger.info(
        "Built sequences for %d patients with >= %d studies (out of %d total patients)",
        len(sequences),
        min_studies,
        len(patient_map),
    )
    return sequences


def build_consecutive_pairs(sequences: dict[str, list[dict]]) -> list[dict]:
    """Create all (prior, current) consecutive study pairs from sequences.

    Each pair is within the same patient and maintains chronological order.
    """
    pairs = []
    for pid, studies in sequences.items():
        for i in range(len(studies) - 1):
            prior = studies[i]
            current = studies[i + 1]
            pairs.append({
                "patient_id": pid,
                "prior_image_id": prior.get("image_id", ""),
                "prior_image_path": prior.get("image_path", ""),
                "prior_study_number": prior.get("followup_metadata", {}).get("study_number", ""),
                "prior_timehash": prior.get("followup_metadata", {}).get("timehash", ""),
                "current_image_id": current.get("image_id", ""),
                "current_image_path": current.get("image_path", ""),
                "current_study_number": current.get("followup_metadata", {}).get("study_number", ""),
                "current_timehash": current.get("followup_metadata", {}).get("timehash", ""),
                "pair_index": i,
            })
    logger.info("Built %d consecutive study pairs", len(pairs))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build GRAZPEDWRI-DX serial X-ray sequences and pairs."
    )
    parser.add_argument("--config", default="model/configs/config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    min_studies = training_cfg.get("phase_5_serial_comparison", {}).get("min_studies_per_patient", 2)

    grazpedwri_path = os.environ.get("GRAZPEDWRI_ROOT", data_cfg.get("grazpedwri_path", ""))
    split_dir = Path(data_cfg.get("split_dir", "model/data/splits"))
    reports_dir = Path(data_cfg.get("reports_dir", "model/reports"))
    pair_cache_dir = Path(data_cfg.get("pair_cache_dir", "model/data/pair_cache"))
    pair_cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    adapter = GRAZPEDWRIAdapter(grazpedwri_path)

    # Load records PER SPLIT to avoid cross-split pairing
    split_records: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        split_csv = split_dir / f"{split_name}.csv"
        if not split_csv.exists():
            logger.warning("Split CSV not found: %s — run create_splits.py first", split_csv)
            continue
        with split_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # Only GRAZPEDWRI records
        split_records[split_name] = [r for r in rows if "GRAZPEDWRI" in r.get("dataset", "")]
        logger.info("Split %s: %d GRAZPEDWRI records", split_name, len(split_records[split_name]))

    if not any(split_records.values()):
        logger.warning(
            "No GRAZPEDWRI records found in split CSVs. "
            "Loading directly from adapter (no split guard — for inspection only)."
        )
        records = adapter.load_records()
        split_records = {"all": records}

    # Enrich split records with followup metadata from adapter
    # Build a lookup from the full adapter records
    all_adapter_records = adapter.load_records()
    id_to_record = {r["image_id"]: r for r in all_adapter_records}

    all_sequences: dict[str, dict[str, list[dict]]] = {}
    all_pairs: dict[str, list[dict]] = {}

    for split_name, raw_records in split_records.items():
        enriched = [id_to_record.get(r.get("image_id", ""), r) for r in raw_records]
        sequences = build_sequences(enriched, min_studies=min_studies)
        pairs = build_consecutive_pairs(sequences)
        all_sequences[split_name] = {
            pid: [{"image_id": r["image_id"], "image_path": r.get("image_path"), "followup_metadata": r.get("followup_metadata", {})} for r in seq]
            for pid, seq in sequences.items()
        }
        all_pairs[split_name] = pairs

        # Write pairs CSV
        pairs_csv = pair_cache_dir / f"{split_name}_pairs.csv"
        if pairs:
            with pairs_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
                writer.writeheader()
                writer.writerows(pairs)
            logger.info("Wrote %d pairs to %s", len(pairs), pairs_csv)

    # Write sequence JSON to reports
    sequences_report = reports_dir / "grazpedwri_sequences.json"
    sequences_report.write_text(
        json.dumps({
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "min_studies_per_patient": min_studies,
            "recovery_labels_available": False,
            "recovery_labels_note": (
                "GRAZPEDWRI-DX has no verified time-to-union / censoring labels. "
                "Supervised recovery-time training is DISABLED."
            ),
            "splits": {
                split_name: {
                    "patient_count": len(seqs),
                    "pair_count": len(all_pairs.get(split_name, [])),
                    "patients": {
                        pid: [s["image_id"] for s in studies]
                        for pid, studies in seqs.items()
                    },
                }
                for split_name, seqs in all_sequences.items()
            },
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\n✓ Sequence and pair data written:")
    print(f"  {sequences_report}")
    for split_name in all_pairs:
        print(f"  {pair_cache_dir / (split_name + '_pairs.csv')}")


if __name__ == "__main__":
    main()
