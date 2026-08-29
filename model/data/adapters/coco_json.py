"""Parse COCO JSON annotation files.

Supports the FracAtlas COCO_fracture_masks.json format which contains
segmentation polygon masks alongside bounding boxes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_coco_json(path: Path) -> dict[str, Any]:
    """Load a COCO JSON file and return a structured dict.

    Returns
    -------
    dict with keys:
        images     : list[dict]  — {id, file_name, width, height}
        categories : list[dict]  — {id, name, supercategory}
        by_image   : dict[str, list[dict]]  — image filename → annotations
            each annotation has: category_id, category_name, bbox_xywh,
            bbox_xyxy, segmentation (list of polygon point lists), area, iscrowd
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    categories: dict[int, str] = {
        cat["id"]: cat["name"] for cat in data.get("categories", [])
    }
    id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in data.get("images", [])
    }

    by_image: dict[str, list[dict]] = {}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        filename = id_to_filename.get(image_id, str(image_id))
        bbox = ann.get("bbox", [])  # COCO format: [x, y, w, h]
        bbox_xyxy = (
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            if len(bbox) == 4
            else []
        )
        record = {
            "annotation_id": ann.get("id"),
            "category_id": ann.get("category_id"),
            "category_name": categories.get(ann.get("category_id", -1), "unknown"),
            "bbox_xywh": bbox,
            "bbox_xyxy": bbox_xyxy,
            "segmentation": ann.get("segmentation", []),  # list of polygon lists
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        }
        by_image.setdefault(filename, []).append(record)

    return {
        "images": data.get("images", []),
        "categories": data.get("categories", []),
        "by_image": by_image,
    }
