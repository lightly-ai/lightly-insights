"""Polygons whose edges are mostly horizontal or vertical.

Signature of "rectangle-trace" labeling: the labeler clicked corners of a
rectangle instead of tracing the object outline. Produces polygons that
are technically polygons but carry no shape information, so training on
them is no better than training on bounding boxes.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Polygon

# Fraction of edges within ~2° of horizontal/vertical above which we flag.
AXIS_ALIGNED_FRACTION = 0.85
# Short polygons aren't meaningful — a triangle is always mostly axis-aligned
# for some rotation. Only flag when there are enough vertices to be suspicious.
MIN_VERTICES = 6


@register_check
class PolygonAxisAlignedCheck(Check):
    check_id = "polygon_axis_aligned"
    title = "Axis-aligned polygon (rectangle-trace labeling)"
    category = "annotation"
    supported_kinds = frozenset({AnnotationKind.POLYGON})

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for ann in dataset.annotations:
            if ann.kind != AnnotationKind.POLYGON:
                continue
            poly = ann.geometry
            if not isinstance(poly, Polygon) or len(poly.points) < MIN_VERTICES:
                continue
            frac = poly.axis_aligned_edge_fraction()
            if frac < AXIS_ALIGNED_FRACTION:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.MEDIUM,
                    category=self.category,
                    title=(
                        f"Rectangle-traced polygon for class '{ann.class_name}'"
                    ),
                    detail=(
                        f"{100 * frac:.0f} % of the polygon's edges on "
                        f"{ann.image_filename} are axis-aligned. "
                        "The labeler likely clicked rectangle corners "
                        "instead of tracing the object outline."
                    ),
                    action=(
                        "Re-label this polygon by following the actual "
                        "object edges. If all classes show this pattern, "
                        "consider using bounding boxes instead."
                    ),
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "axis_aligned_fraction": round(frac, 3),
                        "vertex_count": len(poly.points),
                        "class_name": ann.class_name,
                    },
                )
            )
        return findings
