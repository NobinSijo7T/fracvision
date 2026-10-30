"""SQLAlchemy database models for FracVision.

Tables:
  patients           — de-identified patient records
  imaging_studies    — per-study metadata
  predictions        — model outputs per study
  healing_comparisons — serial X-ray comparison results
  model_versions     — deployed model version tracking
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text,
    create_engine, ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

_DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fracvision_dev.db")
)
DATABASE_URL = os.environ.get(
    "DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH.replace(os.sep, '/') }"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """Dependency for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all tables (safe to call multiple times)."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(64), unique=True, index=True, nullable=False)
    dataset_source = Column(String(32), nullable=False)   # "FracAtlas" | "GRAZPEDWRI-DX"
    age = Column(Float, nullable=True)
    gender = Column(String(8), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    studies = relationship("ImagingStudy", back_populates="patient")


class ImagingStudy(Base):
    __tablename__ = "imaging_studies"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(64), ForeignKey("patients.patient_id"), nullable=False)
    image_id = Column(String(128), unique=True, index=True, nullable=False)
    study_number = Column(String(32), nullable=True)
    timehash = Column(String(32), nullable=True)
    projection = Column(String(32), nullable=True)
    laterality = Column(String(8), nullable=True)
    ao_classification = Column(String(32), nullable=True)
    fracture_visible = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="studies")
    predictions = relationship("Prediction", back_populates="study")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(String(128), ForeignKey("imaging_studies.image_id"), nullable=False)
    model_version = Column(String(64), nullable=True)
    fracture_detected = Column(Boolean, nullable=True)
    fracture_confidence = Column(Float, nullable=True)
    detections_json = Column(JSON, nullable=True)     # list of bbox dicts
    classification_json = Column(JSON, nullable=True)  # labels + probs
    segmentation_json = Column(JSON, nullable=True)   # mask area, score
    qc_passed = Column(Boolean, nullable=True)
    stage_errors_json = Column(JSON, nullable=True)
    clinically_validated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    study = relationship("ImagingStudy", back_populates="predictions")


class HealingComparison(Base):
    __tablename__ = "healing_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(64), index=True, nullable=False)
    prior_image_id = Column(String(128), nullable=True)
    current_image_id = Column(String(128), nullable=True)
    embedding_similarity = Column(Float, nullable=True)
    pixel_mean_abs_diff = Column(Float, nullable=True)
    mask_dice = Column(Float, nullable=True)
    mask_iou = Column(Float, nullable=True)
    registration_applied = Column(Boolean, default=False)
    clinically_validated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(64), nullable=False)
    version = Column(String(32), nullable=False)
    checkpoint_path = Column(Text, nullable=True)
    mlflow_run_id = Column(String(64), nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
