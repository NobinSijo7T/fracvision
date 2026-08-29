"""Mamba longitudinal model training script.

Usage:
  python model/training/train_mamba.py --config model/configs/config.yaml
  python model/training/train_mamba.py --config model/configs/config.yaml --dry-run

Trains on GRAZPEDWRI-DX patient sequences using fracture_visible label
progression as the healing-stage signal.

Recovery-time supervised training is DISABLED (no clinical outcome labels).
This is checked and enforced at runtime.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import yaml

from model.models.mamba.recovery import is_training_enabled, get_disabled_reason

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def build_sequence_batch(
    patient_records: list[dict],
    extractor,
    image_size: int,
    device: torch.device,
    n_det_features: int = 5,
    n_seg_features: int = 3,
    n_clinical: int = 4,
    max_seq_len: int = 10,
    dry_run: bool = False,
):
    """Build tensors for one patient's sequence of studies."""
    from model.data.preprocessing.preprocess import preprocess_image

    records = patient_records[:max_seq_len]
    embeddings, det_feats, seg_feats, time_deltas, clinical_feats = [], [], [], [], []

    base_timehash = None
    for rec in records:
        img_path = rec.get("image_path")
        try:
            if img_path and Path(img_path).exists():
                img, _ = preprocess_image(img_path, image_size=image_size, return_tensor=True,
                                          normalize_mean=[0.5], normalize_std=[0.5])
                t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    feats = extractor.extract(t)
                emb = feats["cls_token"].squeeze(0).cpu()
            else:
                emb = torch.zeros(extractor.embedding_dim)
        except Exception:
            emb = torch.zeros(extractor.embedding_dim)

        embeddings.append(emb)

        # Timehash → days delta
        th = int(rec.get("followup_metadata", {}).get("timehash") or 0)
        if base_timehash is None:
            base_timehash = th
        days = max(0, (th - base_timehash) / 86400)  # seconds → days

        det_feats.append(torch.zeros(n_det_features))
        seg_feats.append(torch.zeros(n_seg_features))
        time_deltas.append(torch.tensor([days]))

        # Clinical features: age (normalised), gender (0/1), cast (0/1), fracture_visible (0/1)
        meta = rec.get("followup_metadata", {}) or rec.get("metadata", {})
        try:
            age = float(meta.get("age") or 0) / 20.0  # rough normalisation
        except (ValueError, TypeError):
            age = 0.0
        gender = 1.0 if str(meta.get("gender", "")).upper() == "M" else 0.0
        cast = 1.0 if str(meta.get("cast", "")).strip() == "1" else 0.0
        fv = 1.0 if "fracture_visible" in (rec.get("labels") or []) else 0.0
        clinical_feats.append(torch.tensor([age, gender, cast, fv]))

    if not embeddings:
        return None

    L = len(embeddings)
    return {
        "embeddings": torch.stack(embeddings).unsqueeze(0),          # (1, L, D)
        "det_features": torch.stack(det_feats).unsqueeze(0),         # (1, L, n_det)
        "seg_features": torch.stack(seg_feats).unsqueeze(0),         # (1, L, n_seg)
        "time_deltas": torch.stack(time_deltas).unsqueeze(0),        # (1, L, 1)
        "clinical_features": torch.stack(clinical_feats).unsqueeze(0), # (1, L, n_clinical)
        "healing_labels": torch.tensor([[int("fracture_visible" in (r.get("labels") or [])) for r in records]], dtype=torch.long), # (1, L)
        "seq_len": L,
    }


def train(config_path: str, dry_run: bool = False) -> None:
    # Enforce recovery-label gate FIRST
    if is_training_enabled():
        raise RuntimeError("Recovery training guard failed — this should never happen.")
    logger.info("Recovery-time supervised training: DISABLED (%s)", get_disabled_reason())

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    mamba_cfg = cfg.get("training", {}).get("phase_6_mamba_longitudinal", {})
    recovery_cfg = cfg.get("training", {}).get("phase_7_recovery", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    seed = cfg.get("seed", 42)

    if recovery_cfg.get("enabled", False):
        logger.error(
            "Recovery training is marked as enabled in config but no clinical labels exist. "
            "This configuration is invalid — recovery training will NOT run."
        )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not mamba_cfg.get("enabled", True):
        logger.info("Mamba longitudinal training disabled in config — skipping.")
        return

    # Load GRAZPEDWRI sequences
    from model.data.adapters import GRAZPEDWRIAdapter
    gz_adapter = GRAZPEDWRIAdapter(os.environ.get("GRAZPEDWRI_ROOT", data_cfg.get("grazpedwri_path", "")))
    all_records = gz_adapter.load_records()
    all_by_id = {r["image_id"]: r for r in all_records}

    split_dir = Path(data_cfg.get("split_dir", "model/data/splits"))
    def load_gz_split(split_name):
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            return []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if "GRAZPEDWRI" in r.get("dataset", "")]
        return [all_by_id.get(r["image_id"], r) for r in rows]

    train_gz = load_gz_split("train")
    val_gz = load_gz_split("val")

    # Build patient sequences
    train_sequences = gz_adapter.get_patient_sequences(train_gz)
    val_sequences = gz_adapter.get_patient_sequences(val_gz)
    min_studies = cfg.get("training", {}).get("phase_5_serial_comparison", {}).get("min_studies_per_patient", 2)
    train_sequences = {k: v for k, v in train_sequences.items() if len(v) >= min_studies}
    val_sequences = {k: v for k, v in val_sequences.items() if len(v) >= min_studies}

    logger.info("Train sequences: %d patients | Val sequences: %d patients", len(train_sequences), len(val_sequences))

    if not train_sequences:
        logger.warning("No multi-study sequences found. Check split CSV and GRAZPEDWRI data.")
        return

    # Load DINOv2
    from model.models.dinov2 import DINOv2Extractor
    dino_cfg_sec = cfg.get("training", {}).get("phase_1_dinov2_adaptation", {})
    extractor = DINOv2Extractor(
        model_name=dino_cfg_sec.get("model_name", "dinov2_vits14"),
        fine_tune_layers=0,
        checkpoint_path=models_cfg.get("dinov2_checkpoint"),
        device=str(device),
    )

    # Build Mamba model
    from model.models.mamba.longitudinal import MambaLongitudinalModel
    mamba = MambaLongitudinalModel(
        d_dino=extractor.embedding_dim,
        n_det_features=5,
        n_seg_features=3,
        n_clinical=4,
        d_model=mamba_cfg.get("mamba_d_model", 256),
        d_state=mamba_cfg.get("mamba_d_state", 16),
        d_conv=mamba_cfg.get("mamba_d_conv", 4),
        expand=mamba_cfg.get("mamba_expand", 2),
        n_layers=mamba_cfg.get("mamba_n_layers", 4),
        n_healing_stages=2,
    ).to(device)

    optimizer = torch.optim.AdamW(mamba.parameters(), lr=mamba_cfg.get("learning_rate", 5e-5))
    criterion = nn.CrossEntropyLoss()

    epochs = mamba_cfg.get("epochs", 40)
    if dry_run:
        epochs = 2

    image_size = cfg.get("preprocessing", {}).get("image_size", 1024)
    ckpt_path = models_cfg.get("mamba_longitudinal_best", "model/checkpoints/mamba_longitudinal_best.pth")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    use_mlflow = False
    try:
        import mlflow
        mlflow.set_experiment(cfg.get("logging", {}).get("mlflow_experiment_name", "fracvision"))
        mlflow.start_run(run_name="mamba_longitudinal")
        mlflow.log_params({"epochs": epochs, "seed": seed, "recovery_enabled": False})
        use_mlflow = True
    except Exception:
        pass

    best_val_loss = float("inf")
    patient_ids = list(train_sequences.keys())
    if dry_run:
        patient_ids = patient_ids[:3]

    for epoch in range(epochs):
        mamba.train()
        ep_loss = 0.0
        for pid in patient_ids:
            batch = build_sequence_batch(
                train_sequences[pid], extractor, image_size, device,
                dry_run=dry_run,
            )
            if batch is None:
                continue
            out = mamba(
                batch["embeddings"].to(device),
                batch["det_features"].to(device),
                batch["seg_features"].to(device),
                batch["time_deltas"].to(device),
                batch["clinical_features"].to(device),
            )
            logits = out["healing_logits"].squeeze(0)  # (L, 2)
            targets = batch["healing_labels"].squeeze(0).to(device)  # (L,)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mamba.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        # Validation
        mamba.eval()
        val_loss = float("inf")
        if val_sequences:
            with torch.no_grad():
                val_losses = []
                for pid in list(val_sequences.keys())[:5]:
                    batch = build_sequence_batch(val_sequences[pid], extractor, image_size, device, dry_run=dry_run)
                    if batch is None:
                        continue
                    out = mamba(batch["embeddings"].to(device), batch["det_features"].to(device),
                                batch["seg_features"].to(device), batch["time_deltas"].to(device),
                                batch["clinical_features"].to(device))
                    logits = out["healing_logits"].squeeze(0)
                    targets = batch["healing_labels"].squeeze(0).to(device)
                    val_losses.append(criterion(logits, targets).item())
                val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(mamba.state_dict(), ckpt_path)

        logger.info("Epoch %d/%d — train_loss: %.4f, val_loss: %.4f", epoch + 1, epochs, ep_loss, val_loss)
        if use_mlflow:
            mlflow.log_metrics({"train_loss": ep_loss, "val_loss": val_loss}, step=epoch)

    if use_mlflow:
        mlflow.end_run()

    print(f"\n✓ Mamba longitudinal training complete. Checkpoint: {ckpt_path}")
    print("  ⚠️  Recovery-time prediction training was NOT performed (no clinical labels).")


def main():
    parser = argparse.ArgumentParser(description="Train Mamba longitudinal model.")
    parser.add_argument("--config", default="model/configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
