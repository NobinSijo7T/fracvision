"""Unit tests for the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pytest


class TestPreprocessImage:
    def test_basic_numpy_input(self):
        from model.data.preprocessing.preprocess import preprocess_image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out, meta = preprocess_image(img, image_size=256)
        assert out.shape == (256, 256, 3), f"Expected (256,256,3), got {out.shape}"
        assert out.dtype == np.uint8
        assert meta["qc_passed"] is True or isinstance(meta["warnings"], list)

    def test_grayscale_input_returns_rgb(self):
        from model.data.preprocessing.preprocess import preprocess_image
        gray = np.random.randint(0, 255, (400, 300), dtype=np.uint8)
        out, meta = preprocess_image(gray, image_size=128)
        assert out.shape == (128, 128, 3), "Grayscale input should be converted to 3-channel output"

    def test_small_image_raises(self):
        from model.data.preprocessing.preprocess import preprocess_image
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            preprocess_image(tiny, image_size=256, min_dimension=64)

    def test_returns_float_tensor(self):
        from model.data.preprocessing.preprocess import preprocess_image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out, meta = preprocess_image(img, image_size=128, return_tensor=True)
        assert out.dtype == np.float32
        assert out.min() >= -5.0 and out.max() <= 5.0, "Float output should be normalised"

    def test_aspect_preserving_pad(self):
        from model.data.preprocessing.preprocess import preprocess_image
        wide = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        out, meta = preprocess_image(wide, image_size=256)
        assert out.shape == (256, 256, 3)

    def test_missing_file_raises(self):
        from model.data.preprocessing.preprocess import preprocess_image
        with pytest.raises(FileNotFoundError):
            preprocess_image("/nonexistent/path/image.jpg")

    def test_real_fracatlas_image(self, sample_fracatlas_image):
        if sample_fracatlas_image is None:
            pytest.skip("No FracAtlas image available")
        from model.data.preprocessing.preprocess import preprocess_image
        out, meta = preprocess_image(str(sample_fracatlas_image), image_size=512)
        assert out.shape == (512, 512, 3)
        assert len(meta["warnings"]) == 0 or isinstance(meta["warnings"], list)

    def test_real_grazpedwri_image(self, sample_grazpedwri_image):
        if sample_grazpedwri_image is None:
            pytest.skip("No GRAZPEDWRI image available")
        from model.data.preprocessing.preprocess import preprocess_image
        out, meta = preprocess_image(str(sample_grazpedwri_image), image_size=512)
        assert out.shape == (512, 512, 3)


class TestAugmentation:
    def test_flip_changes_image(self):
        from model.data.preprocessing.augmentation import random_horizontal_flip
        import random
        img = np.arange(0, 256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        rng = random.Random(0)
        # Force flip by setting p=1.0
        flipped, _ = random_horizontal_flip(img, [], p=1.0, rng=rng)
        assert not np.array_equal(img, flipped), "Horizontal flip should change image"

    def test_bbox_flipped_correctly(self):
        from model.data.preprocessing.augmentation import random_horizontal_flip
        import random
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        bboxes = [{"bbox_xyxy": [10.0, 20.0, 50.0, 80.0]}]
        rng = random.Random(0)
        _, new_bboxes = random_horizontal_flip(img, bboxes, p=1.0, rng=rng)
        # x coords should be mirrored: new_xmin = W - old_xmax = 200 - 50 = 150
        assert abs(new_bboxes[0]["bbox_xyxy"][0] - 150.0) < 1.0

    def test_training_augmentation_passthrough(self):
        from model.data.preprocessing.augmentation import TrainingAugmentation
        aug = TrainingAugmentation(seed=42)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out, bboxes = aug(img, [])
        assert out.shape == img.shape
        assert isinstance(bboxes, list)

    def test_validation_augmentation_noop(self):
        from model.data.preprocessing.augmentation import ValidationAugmentation
        aug = ValidationAugmentation()
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out, _ = aug(img)
        assert np.array_equal(img, out), "Validation augmentation should be a no-op"
