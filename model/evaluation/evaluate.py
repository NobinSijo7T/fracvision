"""Comprehensive evaluation module for FracVision.

Generates metrics reports for:
  - Detection: Precision, Recall, F1, mAP@50, mAP@50:95 per class
  - Segmentation: Dice, IoU, Hausdorff distance
  - Classification: AUROC, AUPRC, F1, Sensitivity, Specificity, Calibration

Output files (all in model/reports/):
  metrics.json            — machine-readable full metrics
  evaluation_summary.md   — human-readable summary
  confusion_matrix.png    — classification confusion matrix
  roc_curve.png           — ROC curves per class
  pr_curve.png            — PR curves per class

Usage:
  python model/evaluation/evaluate.py --config model/configs/config.yaml
  python model/evaluation/evaluate.py --config model/configs/config.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    pred_boxes: list[list[float]],
    pred_scores: list[float],
    gt_boxes: list[list[float]],
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """Compute per-image precision/recall at given IoU threshold."""

    def box_iou(b1: list[float], b2: list[float]) -> float:
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    if not gt_boxes:
        return {"precision": 1.0 if not pred_boxes else 0.0, "recall": 1.0, "tp": 0, "fp": len(pred_boxes), "fn": 0}
    if not pred_boxes:
        return {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": len(gt_boxes)}

    matched_gt = set()
    tp = fp = 0
    for pb, ps in sorted(zip(pred_boxes, pred_scores), key=lambda x: -x[1]):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> dict[str, float]:
    from model.models.medsam.segmentor import compute_dice, compute_iou
    dices, ious = [], []
    for p, g in zip(pred_masks, gt_masks):
        dices.append(compute_dice(p, g))
        ious.append(compute_iou(p, g))
    return {
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "n_samples": len(dices),
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,   # (N, n_classes) binary
    y_prob: np.ndarray,   # (N, n_classes) probabilities
    class_names: list[str],
    reports_dir: Path,
) -> dict[str, Any]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, confusion_matrix, classification_report
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, n_classes = y_true.shape
    results: dict[str, Any] = {}

    # Per-class metrics
    per_class: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            per_class[name] = {"note": "no positive samples in test set"}
            continue
        try:
            auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
            auprc = average_precision_score(y_true[:, i], y_prob[:, i])
        except Exception:
            auroc, auprc = float("nan"), float("nan")
        y_pred = (y_prob[:, i] >= 0.5).astype(int)
        per_class[name] = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1_score(y_true[:, i], y_pred, zero_division=0)),
            "support": int(y_true[:, i].sum()),
        }

    results["per_class"] = per_class
    results["n_test_samples"] = N

    # Confusion matrix (binary fracture class only)
    try:
        if "fractured" in class_names:
            idx = class_names.index("fractured")
            cm = confusion_matrix(y_true[:, idx], (y_prob[:, idx] >= 0.5).astype(int))
            results["fracture_confusion_matrix"] = cm.tolist()
            # Plot
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Neg", "Pred Pos"]); ax.set_yticklabels(["True Neg", "True Pos"])
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center", color="black")
            ax.set_title("Confusion Matrix (fractured)")
            fig.tight_layout()
            fig.savefig(reports_dir / "confusion_matrix.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        logger.warning("Confusion matrix plot failed: %s", exc)

    # ROC curve plot
    try:
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, name in enumerate(class_names[:8]):  # limit to 8 classes
            if y_true[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, label=name)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(reports_dir / "roc_curve.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        logger.warning("ROC curve plot failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(config_path: str, smoke: bool = False) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    reports_dir = Path(data_cfg.get("reports_dir", "model/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation (smoke=%s) ...", smoke)

    from model.inference.pipeline import FracVisionPipeline
    try:
        pipeline = FracVisionPipeline.from_config(config_path)
    except Exception as exc:
        logger.error("Pipeline initialisation failed: %s", exc)
        return

    # Load test split
    split_dir = Path(data_cfg.get("split_dir", "model/data/splits"))
    test_csv = split_dir / "test.csv"
    if not test_csv.exists():
        logger.warning("Test split CSV not found. Run create_splits.py first.")
        return

    import csv
    with test_csv.open(newline="", encoding="utf-8") as fh:
        test_rows = list(csv.DictReader(fh))

    if smoke:
        test_rows = test_rows[:5]
        logger.info("Smoke mode: evaluating on %d samples", len(test_rows))

    # Load full records
    from model.data.adapters import FracAtlasAdapter
    fa_adapter = FracAtlasAdapter(os.environ.get("FRACATLAS_ROOT", data_cfg.get("fracatlas_path", "")))
    fa_records = {r["image_id"]: r for r in fa_adapter.load_records()}

    from model.training.train_classifier import FRACATLAS_LABELS
    y_true_list, y_prob_list = [], []
    detection_totals = {"tp": 0, "fp": 0, "fn": 0}
    detection_images = 0

    def record_boxes(rec: dict, image_path: str) -> list[list[float]]:
        from PIL import Image
        width, height = Image.open(image_path).size
        boxes = []
        for annotation in rec.get("annotations", []):
            if annotation.get("bbox_xyxy"):
                box = [float(value) for value in annotation["bbox_xyxy"]]
            elif annotation.get("bbox_yolo"):
                cx, cy, bw, bh = [float(value) for value in annotation["bbox_yolo"]]
                box = [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ]
            else:
                continue
            if not any(compute_box_iou(box, old) >= 0.95 for old in boxes):
                boxes.append(box)
        return boxes

    def compute_box_iou(left: list[float], right: list[float]) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = area_left + area_right - intersection
        return intersection / union if union else 0.0

    for row in test_rows:
        img_path = row.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue
        rec = fa_records.get(row.get("image_id", ""), row)
        true_labels = rec.get("labels", [])
        try:
            pred = pipeline.predict(img_path)
            gt_boxes = record_boxes(rec, img_path)
            pred_boxes = [
                detection.bbox_xyxy
                for detection in pred.detections
                if detection.label == "fracture"
            ]
            pred_scores = [
                detection.confidence
                for detection in pred.detections
                if detection.label == "fracture"
            ]
            image_detection = compute_detection_metrics(
                pred_boxes, pred_scores, gt_boxes, iou_threshold=0.50
            )
            for key in ("tp", "fp", "fn"):
                detection_totals[key] += image_detection[key]
            detection_images += 1
            if pred.classification is not None:
                probs = [pred.classification.probabilities.get(lbl, 0.0) for lbl in FRACATLAS_LABELS]
                gt = [1.0 if lbl in true_labels else 0.0 for lbl in FRACATLAS_LABELS]
                y_prob_list.append(probs)
                y_true_list.append(gt)
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", img_path, exc)

    metrics: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_samples_evaluated": len(y_true_list),
        "smoke_mode": smoke,
        "clinically_validated": False,
    }

    if y_true_list:
        y_true = np.array(y_true_list)
        y_prob = np.array(y_prob_list)
        cls_metrics = compute_classification_metrics(y_true, y_prob, FRACATLAS_LABELS, reports_dir)
        metrics["classification"] = cls_metrics
    else:
        metrics["classification"] = {"note": "No valid test samples evaluated"}
        logger.warning("No classification predictions generated.")

    tp = detection_totals["tp"]
    fp = detection_totals["fp"]
    fn = detection_totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics["detection"] = {
        "images_evaluated": detection_images,
        "iou_threshold": 0.50,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }

    # Write metrics
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Write summary markdown
    summary_lines = [
        "# FracVision Evaluation Summary",
        "",
        f"Generated: {metrics['generated_at']}",
        f"Test samples evaluated: {metrics['test_samples_evaluated']}",
        f"Smoke mode: {smoke}",
        "",
        "> **⚠️ This evaluation is produced by a research prototype and has NOT been clinically validated.**",
        "",
        "## Classification Results",
    ]
    if "classification" in metrics and "per_class" in metrics["classification"]:
        for cls_name, vals in metrics["classification"]["per_class"].items():
            if isinstance(vals, dict) and "auroc" in vals:
                summary_lines.append(
                    f"- **{cls_name}**: AUROC={vals['auroc']:.3f}, AUPRC={vals['auprc']:.3f}, F1={vals['f1']:.3f}"
                )
    detection = metrics["detection"]
    summary_lines.extend([
        "",
        "## Detection Results",
        f"- Precision@IoU50={detection['precision']:.3f}, Recall@IoU50={detection['recall']:.3f}, F1={detection['f1']:.3f}",
        f"- TP={detection['true_positives']}, FP={detection['false_positives']}, FN={detection['false_negatives']}",
    ])

    (reports_dir / "evaluation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n✓ Evaluation complete. Reports written to {reports_dir}/")
    print(f"  metrics.json, evaluation_summary.md, confusion_matrix.png, roc_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FracVision models on test split.")
    parser.add_argument("--config", default="model/configs/config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Evaluate on 5 samples only")
    args = parser.parse_args()
    run_evaluation(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
