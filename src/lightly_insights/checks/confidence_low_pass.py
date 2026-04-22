"""Low-confidence autolabels that slipped past whatever threshold produced them.

Aggregated one finding per class so a dataset with 20 % low-confidence
annotations doesn't explode into 20k individual findings. The review
queue still fan-outs per affected annotation (every entry in
``affected_annotations`` becomes one queue row), so nothing is lost
for reviewers — the HTML report and the JSON snapshot just stay a
readable size.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Annotation, AnnotationKind, Dataset
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
        return any(a.confidence is not None for a in dataset.annotations)

    def run(self, dataset: Dataset) -> List[Finding]:
        by_class: Dict[str, List[Annotation]] = defaultdict(list)
        for ann in dataset.annotations:
            if ann.confidence is None or ann.confidence >= LOW_CONFIDENCE_THRESHOLD:
                continue
            by_class[ann.class_name].append(ann)

        findings: List[Finding] = []
        for class_name, anns in sorted(by_class.items()):
            mean_conf = sum(a.confidence for a in anns) / len(anns)  # type: ignore[misc]
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.MEDIUM,
                    category=self.category,
                    title=(
                        f"{len(anns)} low-confidence '{class_name}' annotation(s)"
                    ),
                    detail=(
                        f"{len(anns)} annotation(s) of class '{class_name}' "
                        f"have confidence below {LOW_CONFIDENCE_THRESHOLD:.2f} "
                        f"(mean {mean_conf:.2f})."
                    ),
                    action=(
                        "Send the bottom-N to human review, or raise the "
                        "autolabeler's confidence threshold."
                    ),
                    affected_images=sorted({a.image_filename for a in anns}),
                    affected_annotations=sorted(a.annotation_id for a in anns),
                    evidence={
                        "class_name": class_name,
                        "count": len(anns),
                        "mean_confidence": round(mean_conf, 4),
                        "threshold": LOW_CONFIDENCE_THRESHOLD,
                    },
                )
            )
        return findings
