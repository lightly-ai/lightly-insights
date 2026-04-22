"""Pretrained-model proposals with no matching real label = candidate missing labels.

Directional counterpart to ``multi_source_disagreement``. Given a dataset
that contains both "real" annotations (human or autolabeler) and
"proposal" annotations (source starting with ``"proposal:"``, typically
from a frozen pretrained model), this check:

1. For each image, finds proposals that don't overlap any real annotation.
2. Emits a finding per orphan proposal — the highest-value review target
   in a review queue.

We intentionally do NOT emit the mirror case (real annotations without a
proposal), because "the model didn't see it" is rarely actionable — the
model could be wrong, the real annotation could be a rare class the
model wasn't trained on, etc.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.ml.yolo_proposer import is_proposal_source

MATCH_IOU = 0.3  # lower than multi-source because class labels may differ across schemas
HIGH_CONFIDENCE = 0.5


@register_check
class MissingLabelProposalCheck(Check):
    check_id = "missing_label_proposal"
    title = "Pretrained-model proposal with no matching label"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def applies_to(self, dataset: Dataset) -> bool:
        return any(is_proposal_source(a.source) for a in dataset.annotations)

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for filename, anns in dataset.annotations_by_image.items():
            proposals = [a for a in anns if is_proposal_source(a.source)]
            reals = [a for a in anns if not is_proposal_source(a.source)]
            if not proposals:
                continue
            for p in proposals:
                if p.confidence is not None and p.confidence < HIGH_CONFIDENCE:
                    # Low-confidence proposals are noisy; skip to keep the
                    # review queue high-signal.
                    continue
                best_iou = 0.0
                for r in reals:
                    iou = p.tight_box.iou(r.tight_box)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= MATCH_IOU:
                    continue
                conf_str = (
                    f"conf={p.confidence:.2f}" if p.confidence is not None else "no confidence"
                )
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        severity=Severity.HIGH,
                        category="annotation",
                        title=(
                            f"Candidate missing label: '{p.class_name}' "
                            f"({p.source}, {conf_str})"
                        ),
                        detail=(
                            f"{p.source} proposed a '{p.class_name}' on "
                            f"{filename} with {conf_str}, but no real "
                            "annotation overlaps it. Likely a missed label."
                        ),
                        action=(
                            "Send to human review — if the proposal is "
                            "correct, add it as a real annotation."
                        ),
                        affected_images=[filename],
                        affected_annotations=[p.annotation_id],
                        evidence={
                            "proposal_source": p.source,
                            "proposal_class": p.class_name,
                            "confidence": p.confidence,
                            "best_iou_with_real_annotations": round(best_iou, 3),
                        },
                    )
                )
        return findings
