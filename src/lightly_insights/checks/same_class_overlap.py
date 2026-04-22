"""Same-class overlapping annotations that aren't outright duplicates.

``duplicate_annotation`` catches near-identical same-class boxes (IoU ≥ 0.9).
This check fills the gap between "identical" and "separate objects":
significant overlap of one annotation inside another usually means two
touching objects were merged into one, or one was annotated twice with
slightly different extents.

Heuristic: flag when the intersection covers > 70 % of the smaller box.
That's tighter than IoU because an enclosed small object would fail an
IoU threshold but still indicates a merge error.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Box

# Fraction of the smaller box covered by the intersection.
CONTAINMENT_THRESHOLD = 0.7
# Ignore pairs already flagged as duplicates (IoU ≥ 0.9).
DUPLICATE_IOU_THRESHOLD = 0.9


@register_check
class SameClassOverlapCheck(Check):
    check_id = "same_class_overlap"
    title = "Same-class annotations overlap significantly"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for filename, anns in dataset.annotations_by_image.items():
            n = len(anns)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = anns[i], anns[j]
                    if a.class_id != b.class_id:
                        continue
                    box_a = a.tight_box
                    box_b = b.tight_box
                    containment, intersection, iou = _containment(box_a, box_b)
                    if iou >= DUPLICATE_IOU_THRESHOLD:
                        # Let duplicate_annotation own this one.
                        continue
                    if containment < CONTAINMENT_THRESHOLD:
                        continue
                    findings.append(
                        Finding(
                            check_id=self.check_id,
                            severity=Severity.MEDIUM,
                            category=self.category,
                            title=(
                                f"Overlapping '{a.class_name}' annotations"
                            ),
                            detail=(
                                f"Two '{a.class_name}' annotations on "
                                f"{filename} overlap: intersection covers "
                                f"{100 * containment:.0f} % of the smaller "
                                f"box (IoU {iou:.2f}). Usually two "
                                "touching objects merged into one, or the "
                                "same object annotated twice with "
                                "different extents."
                            ),
                            action=(
                                "Inspect; if they're separate objects keep "
                                "both, otherwise delete the redundant box."
                            ),
                            affected_images=[filename],
                            affected_annotations=[a.annotation_id, b.annotation_id],
                            evidence={
                                "containment": round(containment, 3),
                                "iou": round(iou, 3),
                                "class_name": a.class_name,
                            },
                        )
                    )
        return findings


def _containment(a: Box, b: Box) -> "tuple[float, float, float]":
    """Return (intersection / min_area, intersection, IoU).

    ``intersection / min_area`` is the fraction of the smaller box covered;
    this catches enclosed-object cases that IoU would miss.
    """
    x1 = max(a.xmin, b.xmin)
    y1 = max(a.ymin, b.ymin)
    x2 = min(a.xmax, b.xmax)
    y2 = min(a.ymax, b.ymax)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    intersection = (x2 - x1) * (y2 - y1)
    min_area = min(a.area, b.area)
    if min_area <= 0:
        return 0.0, intersection, 0.0
    iou = intersection / max(a.area + b.area - intersection, 1e-9)
    return intersection / min_area, intersection, iou
