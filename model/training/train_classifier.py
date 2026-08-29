"""DINOv2 fracture classification training script.

Usage:
  python model/training/train_classifier.py --config model/configs/config.yaml
  python model/training/train_classifier.py --config model/configs/config.yaml --dry-run

Phase 1: Frozen-feature linear probe baseline
Phase 2: Optional partial fine-tune (top N DINOv2 blocks)

Trains classification head on:
  FracAtlas:       fractured, hand, leg, hip, shoulder, frontal, lateral, oblique
  GRAZPEDWRI-DX:  fracture_visible

Only labels confirmed to exist in the dataset are used.
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# All FracAtlas classification labels (confirmed from dataset.csv inspection)
FRACATLAS_LABELS = [
    "fractured", "hand", "leg", "hip", "shoulder",
    "mixed", "hardware", "multiscan",
    "frontal", "lateral", "oblique",
]


def build_label_tensor(record_labels: list[str], all_labels: list[str]) -> torch.Tensor:
    t = torch.zeros(len(all_labels))
    for lbl in record_labels:
        if lbl in all_labels:
            t[all_labels.index(lbl)] = 1.0
    return t


def generate_embeddings(records, extractor, image_size, device, batch_size=8, dry_run=False):
    """Generate DINOv2 embeddings for all records. Returns (embeddings, label_tensors, ids)."""
    from model.data.preprocessing.preprocess import preprocess_image
    import numpy as np

    embeddings, labels, ids = [], [], []
    n = min(20, len(records)) if dry_run else len(records)

    for i in range(0, n, batch_size):
        batch_records = records[i : i + batch_size]
        batch_tensors = []
        batch_labels = []
        batch_ids = []

        for rec in batch_records:
            img_path = rec.get("image_path")
            if not img_path or not Path(img_path).exists():
                continue
            try:
                img, _ = preprocess_image(img_path, image_size=image_size, return_tensor=True,
                                          normalize_mean=[0.5], normalize_std=[0.5])
                t = torch.from_numpy(img.transpose(2, 0, 1)).float()
                batch_tensors.append(t)
                batch_labels.append(rec.get("labels", []))
                batch_ids.append(rec.get("image_id", ""))
            except Exception as exc:
                logger.warning("Skipping %s: %s", rec.get("image_id", "?"), exc)

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        feats = extractor.extract(batch)
        embeddings.append(feats["cls_token"].cpu())
        labels.extend(batch_labels)
        ids.extend(batch_ids)

        if (i // batch_size) % 10 == 0:
            logger.info("Embedded %d/%d images ...", i + len(batch_tensors), n)

    if not embeddings:
        return torch.zeros(0, extractor.embedding_dim), [], []
    return torch.cat(embeddings, dim=0), labels, ids


def train(config_path: str, dry_run: bool = False) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cls_cfg = cfg.get("training", {}).get("phase_4_classification", {})
    dino_cfg = cfg.get("training", {}).get("phase_1_dinov2_adaptation", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training DINOv2 classifier on %s", device)

    if not cls_cfg.get("enabled", True):
        logger.info("Classification training disabled in config — skipping.")
        return

    # Load adapters
    from model.data.adapters import FracAtlasAdapter, GRAZPEDWRIAdapter
    fa_adapter = FracAtlasAdapter(os.environ.get("FRACATLAS_ROOT", data_cfg.get("fracatlas_path", "")))
    fa_records_full = {r["image_id"]: r for r in fa_adapter.load_records()}

    split_dir = Path(data_cfg.get("split_dir", "model/data/splits"))

    def load_enriched_split(split_name: str, dataset_name: str, all_recs: dict) -> list[dict]:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            logger.warning("Split CSV not found: %s — run create_splits.py first", csv_path)
            return []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if dataset_name in r.get("dataset", "")]
        return [all_recs.get(r["image_id"], r) for r in rows]

    train_records = load_enriched_split("train", "FracAtlas", fa_records_full)
    val_records = load_enriched_split("val", "FracAtlas", fa_records_full)

    if not train_records:
        logger.warning("No training records found. Run create_splits.py first.")
        return

    # Load DINOv2
    from model.models.dinov2 import DINOv2Extractor, FractureClassifier
    model_name = dino_cfg.get("model_name", "dinov2_vits14")
    fine_tune = dino_cfg.get("fine_tune_layers", 0)
    extractor = DINOv2Extractor(
        model_name=model_name,
        fine_tune_layers=fine_tune,
        checkpoint_path=models_cfg.get("dinov2_checkpoint"),
        device=str(device),
    )

    image_size = cfg.get("preprocessing", {}).get("image_size", 1024)
    batch_size = cfg.get("training", {}).get("batch_size", 4)

    logger.info("Generating training embeddings ...")
    train_embs, train_raw_labels, train_ids = generate_embeddings(
        train_records, extractor, image_size, device, batch_size, dry_run
    )
    logger.info("Generating validation embeddings ...")
    val_embs, val_raw_labels, val_ids = generate_embeddings(
        val_records, extractor, image_size, device, batch_size, dry_run
    )

    if train_embs.shape[0] == 0:
        logger.error("No embeddings generated. Check image paths.")
        return

    # Build label tensors
    all_labels = FRACATLAS_LABELS
    train_label_t = torch.stack([build_label_tensor(lbl, all_labels) for lbl in train_raw_labels])
    val_label_t = torch.stack([build_label_tensor(lbl, all_labels) for lbl in val_raw_labels]) if val_embs.shape[0] > 0 else torch.zeros(0, len(all_labels))

    # Classifier
    classifier = FractureClassifier(
        embedding_dim=extractor.embedding_dim,
        class_names=all_labels,
        hidden_dims=[512, 256],
        dropout=0.2,
        use_focal_loss=cls_cfg.get("use_focal_loss", True),
        focal_alpha=cls_cfg.get("focal_alpha", 0.25),
        focal_gamma=cls_cfg.get("focal_gamma", 2.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=cls_cfg.get("learning_rate", 1e-4),
        weight_decay=cls_cfg.get("weight_decay", 1e-4),
    )

    epochs = cls_cfg.get("epochs", 30)
    if dry_run:
        epochs = 2

    # MLflow
    use_mlflow = False
    try:
        import mlflow
        mlflow.set_experiment(cfg.get("logging", {}).get("mlflow_experiment_name", "fracvision"))
        mlflow.start_run(run_name="dinov2_classifier")
        mlflow.log_params({"model": model_name, "epochs": epochs, "classes": all_labels, "seed": seed})
        use_mlflow = True
    except Exception:
        pass

    best_val_loss = float("inf")
    ckpt_path = models_cfg.get("dinov2_classifier_best", "model/checkpoints/dinov2_classifier_best.pth")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        classifier.train()
        train_embs_dev = train_embs.to(device)
        train_label_dev = train_label_t.to(device)

        # Mini-batch training
        perm = torch.randperm(train_embs_dev.shape[0])
        ep_loss = 0.0
        for i in range(0, train_embs_dev.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            loss = classifier.compute_loss(train_embs_dev[idx], train_label_dev[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        # Validation
        classifier.eval()
        val_loss = float("inf")
        if val_embs.shape[0] > 0:
            with torch.no_grad():
                val_loss = classifier.compute_loss(val_embs.to(device), val_label_t.to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(classifier.state_dict(), ckpt_path)

        logger.info("Epoch %d/%d — train_loss: %.4f, val_loss: %.4f", epoch + 1, epochs, ep_loss, val_loss)
        if use_mlflow:
            mlflow.log_metrics({"train_loss": ep_loss, "val_loss": val_loss}, step=epoch)

    # Calibrate on validation set
    if val_embs.shape[0] > 0:
        logger.info("Calibrating temperature scaling on validation set ...")
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embs.to(device))
        classifier.calibrator.calibrate(val_logits, val_label_t.float().to(device))
        torch.save(classifier.state_dict(), ckpt_path)

    if use_mlflow:
        mlflow.end_run()

    print(f"\n✓ Classifier training complete. Best checkpoint: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DINOv2 fracture classifier.")
    parser.add_argument("--config", default="model/configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
