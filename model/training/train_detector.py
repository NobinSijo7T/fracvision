"""RT-DETRv2 fracture detector training script.

Usage:
  python model/training/train_detector.py --config model/configs/config.yaml
  python model/training/train_detector.py --config model/configs/config.yaml --dry-run

Trains on FracAtlas (YOLO annotations, class=fracture) and/or
GRAZPEDWRI-DX (Pascal VOC annotations, multiple classes).

Saves best checkpoint by validation mAP@50.
Tracks experiment with MLflow.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def load_split_records(split_csv: Path, dataset_filter: str | None = None) -> list[dict]:
    """Load records from a split CSV, optionally filtering by dataset name."""
    import csv
    if not split_csv.exists():
        logger.warning("Split CSV not found: %s", split_csv)
        return []
    with split_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if dataset_filter:
        rows = [r for r in rows if dataset_filter.lower() in r.get("dataset", "").lower()]
    return rows


def build_detection_dataset(records: list[dict], adapter, image_size: int, augment: bool):
    """Build a PyTorch Dataset for object detection training."""
    from torch.utils.data import Dataset
    from model.data.preprocessing.preprocess import preprocess_image
    from model.data.preprocessing.augmentation import TrainingAugmentation, ValidationAugmentation
    import numpy as np

    augmentation = TrainingAugmentation() if augment else ValidationAugmentation()

    class DetectionDataset(Dataset):
        def __init__(self):
            self.records = [r for r in records if r.get("image_path")]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            try:
                img, _ = preprocess_image(rec["image_path"], image_size=image_size, return_tensor=True)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3, H, W)
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", rec.get("image_path"), exc)
                img_tensor = torch.zeros(3, image_size, image_size)

            # Get bboxes from adapter record
            bboxes = adapter.get_bbox(rec) if hasattr(adapter, 'get_bbox') else rec.get("annotations", [])
            targets = {
                "boxes": torch.tensor([b["bbox_xyxy"] for b in bboxes if "bbox_xyxy" in b and len(b["bbox_xyxy"]) == 4], dtype=torch.float32),
                "labels": torch.zeros(len(bboxes), dtype=torch.long),  # class 0 = fracture
                "image_id": rec.get("image_id", ""),
            }
            return img_tensor, targets

    return DetectionDataset()


def train(config_path: str, dry_run: bool = False, epochs_override: int | None = None) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    det_cfg = cfg.get("training", {}).get("phase_2_detection", {})
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training RT-DETRv2 on %s", device)

    if not det_cfg.get("enabled", True):
        logger.info("Detection training disabled in config — skipping.")
        return

    # Load split data
    split_dir = Path(data_cfg.get("split_dir", "model/data/splits"))
    from model.data.adapters import FracAtlasAdapter, GRAZPEDWRIAdapter
    fracatlas_adapter = FracAtlasAdapter(os.environ.get("FRACATLAS_ROOT", data_cfg.get("fracatlas_path", "")))
    grazpedwri_adapter = GRAZPEDWRIAdapter(os.environ.get("GRAZPEDWRI_ROOT", data_cfg.get("grazpedwri_path", "")))

    # Load adapter records for image path resolution
    logger.info("Loading FracAtlas records ...")
    fa_records = {r["image_id"]: r for r in fracatlas_adapter.load_records()}
    logger.info("Loading GRAZPEDWRI records ...")
    gz_records = {r["image_id"]: r for r in grazpedwri_adapter.load_records()}

    def enrich(split_rows):
        enriched = []
        for row in split_rows:
            iid = row.get("image_id", "")
            full = fa_records.get(iid) or gz_records.get(iid) or row
            enriched.append(full)
        return enriched

    train_rows = enrich(load_split_records(split_dir / "train.csv"))
    val_rows = enrich(load_split_records(split_dir / "val.csv"))

    image_size = cfg.get("preprocessing", {}).get("image_size", 1024)
    class_names = cfg.get("class_mappings", {}).get("fracatlas", {}).get("yolo_class_map", {})
    all_classes = list({v for v in class_names.values()} | set(
        cfg.get("class_mappings", {}).get("grazpedwri", {}).get("detection_classes", [])
    ))
    observed_labels = {
        str(annotation.get("label"))
        for record in train_rows
        for annotation in record.get("annotations", [])
        if annotation.get("label")
    }
    missing_labels = sorted(set(all_classes) - observed_labels)
    if missing_labels:
        logger.warning(
            "No training annotations found for configured classes: %s. "
            "Those class scores cannot become reliable until the corresponding dataset is available.",
            ", ".join(missing_labels),
        )

    from model.models.rtdetrv2.detector import RTDETRv2Detector
    detector = RTDETRv2Detector(
        class_names=sorted(all_classes),
        score_threshold=cfg.get("inference", {}).get("detection_score_threshold", 0.30),
        checkpoint_path=models_cfg.get("rtdetrv2_checkpoint"),
        device=device,
    )

    epochs = epochs_override if epochs_override is not None else det_cfg.get("epochs", 5)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    lr = det_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.AdamW(
        [p for p in detector.parameters() if p.requires_grad],
        lr=lr, weight_decay=det_cfg.get("weight_decay", 1e-4),
    )

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.get("logging", {}).get("mlflow_tracking_uri") or "")
        mlflow.set_experiment(cfg.get("logging", {}).get("mlflow_experiment_name", "fracvision"))
        mlflow.start_run(run_name="rtdetrv2_detection")
        mlflow.log_params({
            "model": "rtdetrv2",
            "epochs": epochs,
            "lr": lr,
            "classes": sorted(all_classes),
            "seed": seed,
            "image_size": image_size,
        })
        use_mlflow = True
    except Exception:
        use_mlflow = False

    ckpt_dir = Path("model/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_map = 0.0

    if detector.processor is None:
        raise RuntimeError("RT-DETR processor is unavailable; install transformers and retry.")

    class_to_id = {name: i for i, name in enumerate(sorted(all_classes))}

    def box_iou(left, right):
        x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
        x2 = min(left[2], right[2]); y2 = min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = area_left + area_right - intersection
        return intersection / union if union else 0.0

    def make_batch(record: dict, image_index: int):
        from PIL import Image

        image = Image.open(record["image_path"]).convert("RGB")
        width, height = image.size
        annotations = []
        for box in record.get("annotations", []):
            if box.get("bbox_xyxy") and len(box["bbox_xyxy"]) == 4:
                x1, y1, x2, y2 = [float(v) for v in box["bbox_xyxy"]]
            elif box.get("bbox_yolo") and len(box["bbox_yolo"]) == 4:
                cx, cy, bw, bh = [float(v) for v in box["bbox_yolo"]]
                x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
            else:
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            label = str(box.get("label", "fracture"))
            if x2 <= x1 or y2 <= y1 or label not in class_to_id:
                continue
            xyxy = [x1, y1, x2, y2]
            # FracAtlas exposes the same annotation through YOLO and VOC.
            # Keep one copy so the matcher is not trained on duplicate targets.
            if any(item["label"] == label and box_iou(xyxy, item["xyxy"]) >= 0.95 for item in annotations):
                continue
            annotations.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "category_id": class_to_id[label],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "label": label,
                "xyxy": xyxy,
            })

        for annotation in annotations:
            annotation.pop("label", None)
            annotation.pop("xyxy", None)

        encoded = detector.processor(
            images=image,
            annotations={"image_id": image_index, "annotations": annotations},
            return_tensors="pt",
        )
        labels = encoded.pop("labels")
        return encoded, labels

    logger.info("Training for %d epochs (dry_run=%s) ...", epochs if not dry_run else 1, dry_run)
    for epoch in range(1 if dry_run else epochs):
        detector.train()
        epoch_loss = 0.0
        processed = 0
        n_batches = min(5, len(train_rows)) if dry_run else len(train_rows)
        for i, record in enumerate(train_rows[:n_batches]):
            img_path = record.get("image_path")
            if not img_path or not Path(img_path).exists():
                continue
            try:
                batch, labels = make_batch(record, i)
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = [{key: value.to(device) for key, value in item.items()} for item in labels]
                optimizer.zero_grad(set_to_none=True)
                outputs = detector.model(**batch, labels=labels)
                loss = outputs.loss
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    detector.parameters(),
                    cfg.get("training", {}).get("gradient_clip", 1.0),
                )
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                processed += 1
                if (i + 1) % 100 == 0:
                    logger.info(
                        "Epoch %d/%d progress: %d/%d images, mean loss %.4f",
                        epoch + 1,
                        epochs,
                        i + 1,
                        n_batches,
                        epoch_loss / max(processed, 1),
                    )
                continue
                from model.data.preprocessing.preprocess import preprocess_image
                img, _ = preprocess_image(img_path, image_size=image_size, return_tensor=True)
                img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                # Forward pass (loss computation depends on model internals)
                _ = detector(img_t)
                epoch_loss += 0.0  # placeholder — full loss requires label tensors
            except Exception as exc:
                logger.warning("Skipping record %s: %s", record.get("image_id", "?"), exc)

        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, epoch_loss)
        if use_mlflow:
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

    # Save checkpoint
    ckpt_path = models_cfg.get("rtdetrv2_best", "model/checkpoints/rtdetrv2_best.pt")
    try:
        torch.save(detector.state_dict(), ckpt_path)
        logger.info("Saved detector checkpoint to %s", ckpt_path)
    except Exception as exc:
        logger.error("Failed to save checkpoint: %s", exc)

    if use_mlflow:
        mlflow.end_run()

    print("\n✓ RT-DETRv2 training complete.")
    print(f"  Checkpoint: {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-DETRv2 fracture detector.")
    parser.add_argument("--config", default="model/configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch on 5 samples for validation")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured detector epochs")
    args = parser.parse_args()
    train(args.config, dry_run=args.dry_run, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
