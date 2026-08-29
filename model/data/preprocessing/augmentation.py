"""Training-time augmentation pipeline with bounding-box-safe transforms.

Augmentations are applied consistently to both the image and its annotations
(bounding boxes, YOLO format, or absolute pixel coordinates).

All transforms use deterministic seeding when a seed is provided.
"""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Bounding-box transform helpers
# ---------------------------------------------------------------------------

def _clip_bbox_xyxy(bbox: list[float], w: int, h: int) -> list[float] | None:
    """Clip an [xmin, ymin, xmax, ymax] bbox to image bounds.
    Returns None if the box becomes degenerate after clipping.
    """
    xmin = max(0.0, min(bbox[0], w))
    ymin = max(0.0, min(bbox[1], h))
    xmax = max(0.0, min(bbox[2], w))
    ymax = max(0.0, min(bbox[3], h))
    if xmax <= xmin or ymax <= ymin:
        return None
    return [xmin, ymin, xmax, ymax]


def _flip_h_bbox(bbox: list[float], img_w: int) -> list[float]:
    xmin, ymin, xmax, ymax = bbox
    return [img_w - xmax, ymin, img_w - xmin, ymax]


def _flip_v_bbox(bbox: list[float], img_h: int) -> list[float]:
    xmin, ymin, xmax, ymax = bbox
    return [xmin, img_h - ymax, xmax, img_h - ymin]


def _rotate_bbox(bbox: list[float], angle_deg: float, img_h: int, img_w: int) -> list[float] | None:
    """Rotate a bbox by angle_deg around image centre. Returns xyxy bbox."""
    cx, cy = img_w / 2, img_h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    xmin, ymin, xmax, ymax = bbox
    corners = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
    ], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])
    rotated = (M @ corners_h.T).T
    x_min_rot, y_min_rot = rotated[:, 0].min(), rotated[:, 1].min()
    x_max_rot, y_max_rot = rotated[:, 0].max(), rotated[:, 1].max()
    return _clip_bbox_xyxy([x_min_rot, y_min_rot, x_max_rot, y_max_rot], img_w, img_h)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def random_horizontal_flip(
    image: np.ndarray,
    bboxes: list[dict[str, Any]],
    p: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = rng or random.Random()
    if rng.random() >= p:
        return image, bboxes
    h, w = image.shape[:2]
    flipped = cv2.flip(image, 1)
    new_bboxes = []
    for box in bboxes:
        b = box.copy()
        if "bbox_xyxy" in b and b["bbox_xyxy"]:
            b["bbox_xyxy"] = _flip_h_bbox(b["bbox_xyxy"], w)
        new_bboxes.append(b)
    return flipped, new_bboxes


def random_rotation(
    image: np.ndarray,
    bboxes: list[dict[str, Any]],
    max_angle: float = 15.0,
    p: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = rng or random.Random()
    if rng.random() >= p:
        return image, bboxes
    h, w = image.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    new_bboxes = []
    for box in bboxes:
        b = box.copy()
        if "bbox_xyxy" in b and b["bbox_xyxy"]:
            rb = _rotate_bbox(b["bbox_xyxy"], angle, h, w)
            b["bbox_xyxy"] = rb if rb else b["bbox_xyxy"]
        new_bboxes.append(b)
    return rotated, new_bboxes


def random_brightness_contrast(
    image: np.ndarray,
    brightness_delta: float = 0.2,
    contrast_delta: float = 0.2,
    p: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Brightness and contrast jitter (image only, bboxes unchanged)."""
    rng = rng or random.Random()
    if rng.random() >= p:
        return image, []
    alpha = 1.0 + rng.uniform(-contrast_delta, contrast_delta)   # contrast
    beta = rng.uniform(-brightness_delta * 255, brightness_delta * 255)  # brightness
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted, []


def random_gamma(
    image: np.ndarray,
    gamma_range: tuple[float, float] = (0.7, 1.4),
    p: float = 0.4,
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    if rng.random() >= p:
        return image
    gamma = rng.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def random_gaussian_noise(
    image: np.ndarray,
    std_range: tuple[float, float] = (1.0, 8.0),
    p: float = 0.3,
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    if rng.random() >= p:
        return image
    std = rng.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


# ---------------------------------------------------------------------------
# Composed augmentation pipeline
# ---------------------------------------------------------------------------

class TrainingAugmentation:
    """Composed augmentation pipeline for training images.

    All spatial transforms are applied to both image and bboxes.
    Pixel-level transforms (brightness, gamma, noise) are image-only.
    """

    def __init__(
        self,
        flip_p: float = 0.5,
        rotation_p: float = 0.5,
        max_rotation: float = 15.0,
        brightness_p: float = 0.5,
        gamma_p: float = 0.4,
        noise_p: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.flip_p = flip_p
        self.rotation_p = rotation_p
        self.max_rotation = max_rotation
        self.brightness_p = brightness_p
        self.gamma_p = gamma_p
        self.noise_p = noise_p
        self._rng = random.Random(seed)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        bboxes = bboxes or []

        # Spatial transforms (bbox-safe)
        image, bboxes = random_horizontal_flip(image, bboxes, self.flip_p, self._rng)
        image, bboxes = random_rotation(image, bboxes, self.max_rotation, self.rotation_p, self._rng)

        # Pixel transforms (image only)
        image, _ = random_brightness_contrast(image, p=self.brightness_p, rng=self._rng)
        image = random_gamma(image, p=self.gamma_p, rng=self._rng)
        image = random_gaussian_noise(image, p=self.noise_p, rng=self._rng)

        return image, bboxes


class ValidationAugmentation:
    """No-op augmentation for validation/test (passthrough only)."""

    def __call__(
        self,
        image: np.ndarray,
        bboxes: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        return image, bboxes or []
