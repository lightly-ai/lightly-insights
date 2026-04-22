"""Core domain model: Dataset, Annotation, Finding, Check.

This package is the stable surface for downstream consumers (LightlyStudio,
CI pipelines, custom reporters). Everything below is a pure-Python API; no
matplotlib, no Jinja, no filesystem side effects.
"""
from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
)
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Box, Mask, Polygon
from lightly_insights.core.health import HealthScore, Subscore, compute_health
from lightly_insights.core.registry import list_checks, run_all
from lightly_insights.core.review_queue import (
    ReviewItem,
    build_review_queue,
    export_review_queue_csv,
)

__all__ = [
    "Annotation",
    "AnnotationKind",
    "Box",
    "Category",
    "Check",
    "Dataset",
    "Finding",
    "HealthScore",
    "Image",
    "Mask",
    "Polygon",
    "ReviewItem",
    "Severity",
    "Subscore",
    "build_review_queue",
    "compute_health",
    "export_review_queue_csv",
    "list_checks",
    "register_check",
    "run_all",
]
