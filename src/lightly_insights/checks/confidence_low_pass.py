"""Low-confidence autolabels that slipped past whatever threshold produced them.

If the autolabel pipeline was supposed to filter at e.g. 0.5 but some
0.2-confidence annotations made it through, that's usually a threshold-
misconfiguration bug. Flag them so they can be removed or re-reviewed.

We also emit a finding when a small fraction of the dataset carries
confidence while most doesn't — a telltale sign that a batch from a
different pipeline got mixed in.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

LOW_CONFIDENCE_THRESHOLD = 0.3


@register_check
class ConfidenceLowPassCheck(Check):
    check_id = "confidence_low_pass"
    title = "Low-confidence autolabel annotations"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def applies_to(self, dataset: Dataset) -> bool:
        # Only relevant when the dataset actually carries confidence scores.
        return any(a.confidence is not None for a in dataset.annotations)

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for ann in dataset.annotations:
            if ann.confidence is None:
                continue
            if ann.confidence >= LOW_CONFIDENCE_THRESHOLD:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.MEDIUM,
                    category=self.category,
                    title=(
                        f"Low-confidence '{ann.class_name}' annotation "
                        f"(conf={ann.confidence:.2f})"
                    ),
                    detail=(
                        f"Annotation on {ann.image_filename} has confidence "
                        f"{ann.confidence:.2f}, below the review threshold "
                        f"of {LOW_CONFIDENCE_THRESHOLD:.2f}."
                    ),
                    action=(
                        "Send to human review, or raise the autolabeler's "
                        "confidence threshold."
                    ),
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "confidence": round(ann.confidence, 4),
                        "class_name": ann.class_name,
                        "source": ann.source,
                    },
                )
            )
        return findings
