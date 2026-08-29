from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc(path: Path) -> tuple[str | None, list[dict]]:
    """Parse only valid VOC boxes; retain source path for auditability."""
    root = ET.parse(path).getroot()
    filename = root.findtext("filename")
    boxes: list[dict] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            xmin, ymin, xmax, ymax = (float(box.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append({"label": obj.findtext("name", "unknown"), "bbox_xyxy": [xmin, ymin, xmax, ymax], "source": str(path)})
    return filename, boxes
