"""Autolabelers are often overconfident on tiny objects.

A detector trained on big cars will happily produce a 0.95-confidence
"car" for a 5-pixel blob on a garbage bin. Humans looking at the same
blob would hesitate. Flag ``confidence ≥ HIGH`` annotations whose
relative area is below ``TINY`` — the highest-value review targets.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

HIGH_CONFIDENCE = 0.9
TINY_REL_AREA = 0.005  # same cutoff as ``analyze.TINY_OBJECT_REL_AREA``


@register_check
class ConfidenceSizeMismatchCheck(Check):
    check_id = "confidence_size_mismatch"
    title = "Over-confident tiny autolabel"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def applies_to(self, dataset: Dataset) -> bool:
        return any(a.confidence is not None for a in dataset.annotations)

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        image_by_name = dataset.image_by_filename
        for ann in dataset.annotations:
            if ann.confidence is None or ann.confidence < HIGH_CONFIDENCE:
                continue
            image = image_by_name.get(ann.image_filename)
            if image is None or image.area <= 0:
                continue
            rel_area = ann.area / image.area
            if rel_area >= TINY_REL_AREA:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.HIGH,
                    category=self.category,
                    title=(
                        f"Over-confident tiny '{ann.class_name}' "
                        f"(conf={ann.confidence:.2f}, {100 * rel_area:.2f}% area)"
                    ),
                    detail=(
                        f"The autolabeler returned {ann.confidence:.2f} "
                        "confidence on an annotation that covers "
                        f"{100 * rel_area:.3f} % of the image on "
                        f"{ann.image_filename}. Overconfidence on tiny "
                        "objects is a common autolabeler failure mode."
                    ),
                    action=(
                        "Send to human review. High-priority for the "
                        "review queue."
                    ),
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "confidence": round(ann.confidence, 4),
                        "rel_area": round(rel_area, 6),
                        "class_name": ann.class_name,
                        "source": ann.source,
                    },
                )
            )
        return findings
