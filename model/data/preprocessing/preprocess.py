"""Image preprocessing pipeline for FracVision.

Handles PNG / JPEG / DICOM input and produces a normalised numpy array
suitable for DINOv2, RT-DETRv2, and MedSAM inputs.

Outputs:
  preprocessed_array : np.ndarray — shape (H, W, 3) uint8 in [0, 255]
  quality_metadata   : dict       — dimensions, mode, qc flags
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}


# ---------------------------------------------------------------------------
# Quality check helpers
# ---------------------------------------------------------------------------

def _compute_blur_score(gray: np.ndarray) -> float:
    """Return Laplacian variance as a blur measure (lower = more blurred)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_exposure_stats(gray: np.ndarray) -> dict[str, float]:
    """Mean and std of pixel intensities as a simple exposure check."""
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "min": float(gray.min()),
        "max": float(gray.max()),
    }


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def _load_dicom(path: Path) -> np.ndarray:
    """Load a DICOM file and return a uint8 grayscale numpy array.

    De-identification: no patient-identifying fields are retained in the
    returned array. Metadata is extracted separately by the caller if needed.
    """
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM files. "
            "Install with: pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(str(path))
    pixel_array = apply_voi_lut(ds.pixel_array, ds)

    # Normalise to uint8
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > pmin:
        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    # Handle various DICOM photometric interpretations
    if len(pixel_array.shape) == 2:
        pass  # grayscale — desired
    elif len(pixel_array.shape) == 3 and pixel_array.shape[2] in (3, 4):
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {pixel_array.shape}")

    return pixel_array


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(
    source: Path | str | bytes | np.ndarray,
    image_size: int = 1024,
    grayscale: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
    normalize_mean: list[float] | None = None,
    normalize_std: list[float] | None = None,
    min_dimension: int = 64,
    return_tensor: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess a single X-ray image.

    Parameters
    ----------
    source        : file path, raw bytes, or already-loaded numpy array
    image_size    : target output size in pixels (square, aspect-preserving pad)
    grayscale     : convert to single-channel if True
    clahe_clip_limit : CLAHE clip limit for contrast normalisation
    clahe_tile_grid  : CLAHE tile grid size
    normalize_mean   : channel-wise mean for final normalisation (None = skip)
    normalize_std    : channel-wise std for final normalisation (None = skip)
    min_dimension    : reject images smaller than this (raises ValueError)
    return_tensor    : if True, return a float32 tensor in [0,1] instead of uint8

    Returns
    -------
    (preprocessed_array, quality_metadata)
    preprocessed_array : uint8 np.ndarray (H, W, 3) or float32 if return_tensor
    quality_metadata   : dict with qc fields
    """
    quality_meta: dict[str, Any] = {"source_type": None, "warnings": []}

    # --- 1. Load raw array ---
    if isinstance(source, np.ndarray):
        raw = source.copy()
        quality_meta["source_type"] = "numpy"
    elif isinstance(source, bytes):
        quality_meta["source_type"] = "bytes"
        try:
            arr = np.frombuffer(source, dtype=np.uint8)
            raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as exc:
            raise ValueError(f"Failed to decode image bytes: {exc}") from exc
    else:
        path = Path(source)
        quality_meta["source_type"] = "file"
        quality_meta["filename"] = path.name

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if path.suffix.lower() == ".dcm":
            raw = _load_dicom(path)
        else:
            try:
                pil_img = Image.open(path)
                pil_img.verify()
                pil_img = Image.open(path)  # reopen after verify
                raw = np.array(pil_img)
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Cannot open image {path.name}: {exc}") from exc

    if raw is None or raw.size == 0:
        raise ValueError("Loaded image is empty.")

    quality_meta["original_shape"] = raw.shape

    # --- 2. Colour-space normalisation ---
    if len(raw.shape) == 2:
        # Already grayscale
        gray = raw.astype(np.uint8)
    elif raw.shape[2] == 4:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if h < min_dimension or w < min_dimension:
        raise ValueError(
            f"Image too small ({w}×{h}). Minimum dimension: {min_dimension}px"
        )

    quality_meta["original_hw"] = [h, w]

    # --- 3. CLAHE contrast normalisation ---
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_grid, clahe_tile_grid),
    )
    enhanced = clahe.apply(gray)

    # --- 4. Quality checks ---
    blur_score = _compute_blur_score(enhanced)
    exposure = _compute_exposure_stats(enhanced)
    quality_meta["blur_score"] = blur_score
    quality_meta["exposure"] = exposure

    BLUR_THRESHOLD = 10.0
    if blur_score < BLUR_THRESHOLD:
        quality_meta["warnings"].append(
            f"Potentially blurry image (Laplacian variance={blur_score:.2f} < {BLUR_THRESHOLD})"
        )
        logger.warning("Low blur score (%.2f) for %s", blur_score, quality_meta.get("filename", "buffer"))

    # --- 5. Aspect-preserving resize with padding ---
    scale = image_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    pad_top = (image_size - new_h) // 2
    pad_bottom = image_size - new_h - pad_top
    pad_left = (image_size - new_w) // 2
    pad_right = image_size - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )
    quality_meta["output_hw"] = [image_size, image_size]
    quality_meta["scale_factor"] = scale
    quality_meta["padding"] = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right}

    # --- 6. Convert to 3-channel (models expect RGB) ---
    if grayscale:
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    else:
        rgb = padded if len(padded.shape) == 3 else cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)

    # --- 7. Optional float normalisation (for model input tensors) ---
    if return_tensor:
        out = rgb.astype(np.float32) / 255.0
        if normalize_mean and normalize_std:
            mean = np.array(normalize_mean, dtype=np.float32)
            std = np.array(normalize_std, dtype=np.float32)
            # Handle single-channel mean/std for grayscale-as-RGB
            if mean.shape == (1,):
                mean = np.repeat(mean, 3)
                std = np.repeat(std, 3)
            out = (out - mean) / std
    else:
        out = rgb.astype(np.uint8)

    quality_meta["qc_passed"] = len(quality_meta["warnings"]) == 0
    return out, quality_meta


def preprocess_from_config(
    source: Path | str | bytes | np.ndarray,
    config: dict[str, Any],
    return_tensor: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience wrapper that reads preprocessing params from config dict."""
    pp = config.get("preprocessing", {})
    return preprocess_image(
        source,
        image_size=pp.get("image_size", 1024),
        grayscale=pp.get("grayscale", True),
        clahe_clip_limit=pp.get("clahe_clip_limit", 2.0),
        clahe_tile_grid=pp.get("clahe_tile_grid", 8),
        normalize_mean=pp.get("normalize_mean"),
        normalize_std=pp.get("normalize_std"),
        min_dimension=pp.get("min_dimension", 64),
        return_tensor=return_tensor,
    )
