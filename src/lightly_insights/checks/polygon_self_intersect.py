"""Polygons whose edges cross themselves.

Most training pipelines rasterize self-intersecting polygons to a
malformed mask silently — the loss never notices and the model learns
noise. Catch them here.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Polygon


@register_check
class PolygonSelfIntersectCheck(Check):
    check_id = "polygon_self_intersect"
    title = "Self-intersecting polygon"
    category = "annotation"
    supported_kinds = frozenset({AnnotationKind.POLYGON})

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for ann in dataset.annotations:
            if ann.kind != AnnotationKind.POLYGON:
                continue
            poly = ann.geometry
            if not isinstance(poly, Polygon):
                continue
            if not poly.self_intersects():
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.HIGH,
                    category=self.category,
                    title=(
                        f"Self-intersecting polygon for class '{ann.class_name}'"
                    ),
                    detail=(
                        f"Polygon on {ann.image_filename} has crossing "
                        f"edges ({len(poly.points)} vertices). Most "
                        "rasterizers produce a malformed mask silently."
                    ),
                    action=(
                        "Re-draw the polygon without self-crossings, or "
                        "simplify before export."
                    ),
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "vertex_count": len(poly.points),
                        "class_name": ann.class_name,
                    },
                )
            )
        return findings
