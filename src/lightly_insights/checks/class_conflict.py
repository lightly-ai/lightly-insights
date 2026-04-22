"""Detect same-image overlapping annotations from *different* classes.

Works on box, polygon, and mask annotations — the check uses each geometry's
``tight_box`` for IoU. For polygons/masks this is a cheap upper bound;
step-2 can add exact polygon-polygon IoU once we need it.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

IOU_THRESHOLD = 0.5


@register_check
class ClassConflictCheck(Check):
    check_id = "class_conflict"
    title = "Overlapping annotations of different classes"
    category = "annotation"
    # Works on any annotation kind because we use tight_box from each
    # geometry. For polygons/masks this is an upper bound on IoU but still
    # useful as a filter.
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
                    if a.class_id == b.class_id:
                        continue
                    iou = a.tight_box.iou(b.tight_box)
                    if iou < IOU_THRESHOLD:
                        continue
                    findings.append(
                        Finding(
                            check_id=self.check_id,
                            severity=Severity.HIGH,
                            category=self.category,
                            title=(
                                f"'{a.class_name}' vs '{b.class_name}' overlap"
                            ),
                            detail=(
                                f"Annotations in {filename} overlap with "
                                f"IoU {iou:.2f} but belong to different "
                                f"classes ({a.class_name} vs {b.class_name})."
                            ),
                            action=(
                                "Verify which class is correct and delete the other."
                            ),
                            affected_images=[filename],
                            affected_annotations=[a.annotation_id, b.annotation_id],
                            evidence={
                                "iou": round(iou, 3),
                                "class_a": a.class_name,
                                "class_b": b.class_name,
                                "box_a": [
                                    a.tight_box.xmin,
                                    a.tight_box.ymin,
                                    a.tight_box.xmax,
                                    a.tight_box.ymax,
                                ],
                                "box_b": [
                                    b.tight_box.xmin,
                                    b.tight_box.ymin,
                                    b.tight_box.xmax,
                                    b.tight_box.ymax,
                                ],
                            },
                        )
                    )
        return findings
