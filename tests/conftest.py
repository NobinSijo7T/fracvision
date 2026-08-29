"""Pytest configuration and shared fixtures for FracVision tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

CONFIG_PATH = "model/configs/config.yaml"


@pytest.fixture(scope="session")
def config():
    """Load the project config once per test session."""
    return yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fracatlas_root(config) -> Path:
    root = Path(os.environ.get("FRACATLAS_ROOT", config["data"]["fracatlas_path"]))
    return root


@pytest.fixture(scope="session")
def grazpedwri_root(config) -> Path:
    root = Path(os.environ.get("GRAZPEDWRI_ROOT", config["data"]["grazpedwri_path"]))
    return root


@pytest.fixture(scope="session")
def fracatlas_adapter(fracatlas_root):
    from model.data.adapters import FracAtlasAdapter
    return FracAtlasAdapter(fracatlas_root)


@pytest.fixture(scope="session")
def grazpedwri_adapter(grazpedwri_root):
    from model.data.adapters import GRAZPEDWRIAdapter
    return GRAZPEDWRIAdapter(grazpedwri_root)


@pytest.fixture(scope="session")
def fracatlas_records(fracatlas_adapter):
    return fracatlas_adapter.load_records()


@pytest.fixture(scope="session")
def grazpedwri_records(grazpedwri_adapter):
    return grazpedwri_adapter.load_records()


@pytest.fixture
def sample_fracatlas_image(fracatlas_root) -> Path | None:
    """Return the path to the first available FracAtlas image."""
    for subdir in ("Fractured", "Non_fractured"):
        d = fracatlas_root / "images" / subdir
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    return f
    return None


@pytest.fixture
def sample_grazpedwri_image(grazpedwri_root) -> Path | None:
    """Return the path to the first available GRAZPEDWRI PNG image."""
    images_dir = grazpedwri_root / "folder_structure" / "images"
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.suffix.lower() == ".png":
                return f
    return None
