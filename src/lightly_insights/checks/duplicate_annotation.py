"""Same-image, same-class annotations with near-identical geometry.

Legacy ``analyze.py`` caught this for boxes only. Ported to the check
framework so it works on polygons and masks too, via ``tight_box`` IoU
(cheap upper bound; pixel-exact IoU for polygons/masks is step-4+).
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

IOU_THRESHOLD = 0.9


@register_check
class DuplicateAnnotationCheck(Check):
    check_id = "duplicate_annotation"
    title = "Duplicate annotation"
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
                    iou = a.tight_box.iou(b.tight_box)
                    if iou < IOU_THRESHOLD:
                        continue
                    findings.append(
                        Finding(
                            check_id=self.check_id,
                            severity=Severity.HIGH,
                            category=self.category,
                            title=(
                                f"Duplicate '{a.class_name}' annotation"
                            ),
                            detail=(
                                f"Two annotations of class '{a.class_name}' "
                                f"on {filename} overlap with IoU {iou:.2f}."
                            ),
                            action="Delete one of the two.",
                            affected_images=[filename],
                            affected_annotations=[a.annotation_id, b.annotation_id],
                            evidence={
                                "iou": round(iou, 3),
                                "class_name": a.class_name,
                            },
                        )
                    )
        return findings
