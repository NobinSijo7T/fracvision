"""Shared pipeline state singleton.

Avoids global variables in the app module. The pipeline is loaded once
at startup and held here for access by route handlers.
"""
from __future__ import annotations
from typing import Any


class PipelineState:
    pipeline: Any = None  # FracVisionPipeline instance
    load_error: str | None = None

    @classmethod
    def is_ready(cls) -> bool:
        return cls.pipeline is not None
