"""FastAPI backend application for FracVision.

Loads the full inference pipeline on startup and exposes REST endpoints
for X-ray analysis, serial comparison, and patient history.

All responses include mandatory clinical disclaimer fields.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.pipeline_state import PipelineState

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.environ.get("FRACVISION_CONFIG", "model/configs/config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and shut down the inference pipeline."""
    logger.info("Starting FracVision backend — loading models ...")
    try:
        from model.inference.pipeline import FracVisionPipeline
        pipeline = FracVisionPipeline.from_config(_CONFIG_PATH)
        PipelineState.pipeline = pipeline
        logger.info("Pipeline loaded successfully")
    except Exception as exc:
        logger.error("Pipeline load failed: %s", exc)
        PipelineState.pipeline = None
        PipelineState.load_error = str(exc)
    yield
    logger.info("Shutting down FracVision backend")


def create_app() -> FastAPI:
    import yaml
    try:
        cfg = yaml.safe_load(Path(_CONFIG_PATH).read_text())
        allow_origins = cfg.get("backend", {}).get("allow_origins", ["*"])
        max_upload_mb = cfg.get("backend", {}).get("max_upload_mb", 50)
    except Exception:
        allow_origins = ["*"]
        max_upload_mb = 50

    app = FastAPI(
        title="FracVision API",
        description=(
            "Research prototype for X-ray fracture analysis using DINOv2 and Mamba. "
            "NOT FOR CLINICAL USE. All outputs require clinician review."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    return app


app = create_app()
