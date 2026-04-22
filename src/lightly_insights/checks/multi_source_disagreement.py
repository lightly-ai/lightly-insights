"""Disagreement between two annotation sources on the same image.

When a dataset carries annotations from multiple ``source`` values (e.g.
one autolabeler + a human-reviewed subset, or two different autolabelers),
this check surfaces:

- **Unmatched** annotations: present in one source, missing in the other.
  Either a missed label or a hallucination, depending on which source you
  trust.
- **Class disagreement**: same image, overlapping boxes, but the two
  sources assigned different classes. Highest-value review target.

The check runs pairwise over every combination of sources in the dataset.
When a source named ``"human"`` is present we treat it as ground truth
and emit human-vs-X findings only (so reviewers see "the autolabeler
missed this human label" rather than the mirror). With only autolabel
sources, disagreements are symmetric.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import (
    Annotation,
    AnnotationKind,
    Dataset,
)
from lightly_insights.core.finding import Finding, Severity

MATCH_IOU = 0.5  # tight-box IoU threshold to consider two annotations "the same object"


@register_check
class MultiSourceDisagreementCheck(Check):
    check_id = "multi_source_disagreement"
    title = "Sources disagree on annotations"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def applies_to(self, dataset: Dataset) -> bool:
        sources = {a.source for a in dataset.annotations if a.source is not None}
        return len(sources) >= 2

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        sources_in_use = {
            a.source for a in dataset.annotations if a.source is not None
        }
        if len(sources_in_use) < 2:
            return []

        # Determine which pairs to compare. If "human" is present we only
        # compare human ↔ everything else to avoid noise from autolabeler-
        # vs-autolabeler symmetry.
        if "human" in sources_in_use:
            other_sources = sorted(sources_in_use - {"human"})
            pairs = [("human", s) for s in other_sources]
        else:
            pairs = [
                tuple(sorted(p)) for p in combinations(sorted(sources_in_use), 2)
            ]

        for filename, anns in dataset.annotations_by_image.items():
            by_source: Dict[str, List[Annotation]] = {}
            for ann in anns:
                if ann.source is None:
                    continue
                by_source.setdefault(ann.source, []).append(ann)

            for src_a, src_b in pairs:
                findings.extend(
                    _compare_single_image(
                        filename=filename,
                        anns_a=by_source.get(src_a, []),
                        anns_b=by_source.get(src_b, []),
                        src_a=src_a,
                        src_b=src_b,
                        reference_is_a=(src_a == "human"),
                    )
                )
        return findings


def _compare_single_image(
    filename: str,
    anns_a: List[Annotation],
    anns_b: List[Annotation],
    src_a: str,
    src_b: str,
    reference_is_a: bool,
) -> List[Finding]:
    findings: List[Finding] = []

    # Greedy matching: for each annotation in A, pick the highest-IoU
    # unclaimed annotation in B. Matches are used once.
    matched_b: set = set()
    matches: List[Tuple[Annotation, Annotation, float]] = []
    unmatched_a: List[Annotation] = []
    for a in anns_a:
        best_iou = 0.0
        best_b: Optional[Annotation] = None
        for b in anns_b:
            if b.annotation_id in matched_b:
                continue
            iou = a.tight_box.iou(b.tight_box)
            if iou > best_iou:
                best_iou = iou
                best_b = b
        if best_b is not None and best_iou >= MATCH_IOU:
            matched_b.add(best_b.annotation_id)
            matches.append((a, best_b, best_iou))
        else:
            unmatched_a.append(a)

    unmatched_b = [b for b in anns_b if b.annotation_id not in matched_b]

    # Class-disagreement findings on matched pairs.
    for a, b, iou in matches:
        if a.class_id == b.class_id:
            continue
        ref = a if reference_is_a else None
        other = b if reference_is_a else None
        findings.append(
            Finding(
                check_id=MultiSourceDisagreementCheck.check_id,
                severity=Severity.HIGH,
                category="annotation",
                title=(
                    f"{src_a}/{src_b} disagree on class "
                    f"({a.class_name} vs {b.class_name})"
                ),
                detail=(
                    f"On {filename}, {src_a} called this "
                    f"'{a.class_name}' while {src_b} called it "
                    f"'{b.class_name}' (IoU {iou:.2f})."
                    + (
                        f" The human label '{ref.class_name}' is the "
                        "reference."
                        if reference_is_a and ref is not None
                        else ""
                    )
                ),
                action=(
                    "Send to human review — decide which class is correct."
                ),
                affected_images=[filename],
                affected_annotations=[a.annotation_id, b.annotation_id],
                evidence={
                    "source_a": src_a,
                    "source_b": src_b,
                    "class_a": a.class_name,
                    "class_b": b.class_name,
                    "iou": round(iou, 3),
                    "confidence_a": a.confidence,
                    "confidence_b": b.confidence,
                },
            )
        )

    # Unmatched findings.
    for ann in unmatched_a:
        findings.append(_unmatched_finding(ann, filename, src_a, src_b, reference_is_a))
    for ann in unmatched_b:
        # When reference_is_a (source A is human), B-unmatched = autolabel
        # hallucination. When symmetric, still worth reporting.
        findings.append(_unmatched_finding(ann, filename, src_b, src_a, False))

    return findings


def _unmatched_finding(
    ann: Annotation,
    filename: str,
    owner: str,
    other: str,
    owner_is_reference: bool,
) -> Finding:
    if owner_is_reference:
        # Human has a label, autolabeler missed it.
        title = (
            f"{other} missed a '{ann.class_name}' that {owner} annotated"
        )
        detail = (
            f"{owner} labeled a '{ann.class_name}' on {filename} but "
            f"{other} produced no matching annotation. Either "
            f"{other} missed the object, or its confidence threshold "
            "is too high."
        )
        action = (
            f"Lower {other}'s threshold, or re-run the autolabeler "
            "on this image."
        )
    else:
        title = (
            f"{owner} has a '{ann.class_name}' that {other} doesn't"
        )
        detail = (
            f"{owner} labeled a '{ann.class_name}' on {filename} but "
            f"{other} has no matching annotation. Could be a real "
            f"object {other} missed, or a hallucination from {owner}."
        )
        action = "Send to human review to decide."
    return Finding(
        check_id=MultiSourceDisagreementCheck.check_id,
        severity=Severity.HIGH if owner_is_reference else Severity.MEDIUM,
        category="annotation",
        title=title,
        detail=detail,
        action=action,
        affected_images=[filename],
        affected_annotations=[ann.annotation_id],
        evidence={
            "owner_source": owner,
            "other_source": other,
            "class_name": ann.class_name,
            "confidence": ann.confidence,
        },
    )
