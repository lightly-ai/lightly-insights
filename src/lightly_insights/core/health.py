"""Findings-aware dataset health score.

Single opinionated 0-100 number plus four subscores so reviewers can see
the big picture at a glance. All inputs come from ``Dataset`` +
``List[Finding]``; the score is pure-function and reproducible.

The model is:

- **Pipeline integrity** (weight 25 %): whether any CRITICAL finding
  fired (corrupt images, degenerate / out-of-bounds / size-mismatched
  annotations). A single critical issue drops this subscore hard.
- **Annotation quality** (weight 35 %): fraction of annotations NOT
  touched by an annotation-category finding. Tuned so a dataset with
  no issues scores 100, half-flagged scores 50.
- **Balance & data coverage** (weight 20 %): penalized by
  ``starved_class`` findings, orphan classes, and background scarcity.
- **Autolabel trust** (weight 20 %): only when confidences exist —
  penalized by ``confidence_low_pass`` / ``confidence_size_mismatch``
  / ``confidence_class_bias`` findings. When no confidences exist this
  subscore is dropped from the weighted average.

Grades: A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F < 60.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity


SUBSCORE_WEIGHTS = {
    "pipeline": 0.25,
    "annotation": 0.35,
    "balance": 0.20,
    "autolabel": 0.20,
}


@dataclass(frozen=True)
class Subscore:
    name: str  # human-readable, e.g. "Annotation quality"
    key: str  # stable id, e.g. "annotation"
    score: float  # 0-100, or -1 for "N/A"
    grade: str  # "A".."F" or "—"
    detail: str  # one-line explanation rendered next to the score


@dataclass(frozen=True)
class HealthScore:
    overall: Optional[float]  # 0-100; None when no subscore was computable
    grade: str
    subscores: List[Subscore]
    issues: List[str]  # up to 5 highest-priority issue bullets


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_health(
    dataset: Dataset, findings: Sequence[Finding]
) -> HealthScore:
    """Roll a dataset + its findings into a 0-100 health read."""
    by_check: dict = {}
    for f in findings:
        by_check.setdefault(f.check_id, []).append(f)

    subscores: List[Subscore] = []

    # ---- Pipeline integrity ----
    critical = [f for f in findings if f.severity < Severity.HIGH]
    # One critical finding = -30 points, two = -60, etc.; floored at 0.
    pipeline_score = max(0.0, 100.0 - 30.0 * len(critical))
    pipeline_detail = (
        "No pipeline-breaking issues."
        if not critical
        else f"{len(critical)} critical issue(s): {', '.join(sorted(set(f.check_id for f in critical)))}."
    )
    subscores.append(
        Subscore(
            name="Pipeline integrity",
            key="pipeline",
            score=pipeline_score,
            grade=_letter_grade(pipeline_score),
            detail=pipeline_detail,
        )
    )

    # ---- Annotation quality ----
    total_ann = dataset.num_annotations
    if total_ann > 0:
        affected_ids: set = set()
        for f in findings:
            if f.category == "annotation":
                affected_ids.update(f.affected_annotations)
        ann_score = max(0.0, 100.0 * (1.0 - len(affected_ids) / total_ann))
        ann_detail = (
            f"{len(affected_ids)} / {total_ann} annotations flagged "
            f"({100 * len(affected_ids) / total_ann:.1f} %)"
        )
    else:
        ann_score = -1.0
        ann_detail = "No annotations."
    subscores.append(
        Subscore(
            name="Annotation quality",
            key="annotation",
            score=ann_score,
            grade=_letter_grade(ann_score) if ann_score >= 0 else "—",
            detail=ann_detail,
        )
    )

    # ---- Balance & data coverage ----
    balance_penalty = 0.0
    balance_reasons: List[str] = []
    for f in by_check.get("starved_class", []):
        balance_penalty += 15.0
        if "class_name" in f.evidence:
            balance_reasons.append(str(f.evidence["class_name"]))
    if by_check.get("background_scarcity"):
        balance_penalty += 10.0
        balance_reasons.append("few backgrounds")
    if by_check.get("split_purity"):
        balance_penalty += 15.0
        balance_reasons.append("split drift")
    if by_check.get("split_leakage"):
        balance_penalty += 30.0
        balance_reasons.append("split leakage")
    balance_score = max(0.0, 100.0 - balance_penalty)
    balance_detail = (
        "No balance issues."
        if not balance_reasons
        else "; ".join(balance_reasons[:4])
    )
    subscores.append(
        Subscore(
            name="Balance & coverage",
            key="balance",
            score=balance_score,
            grade=_letter_grade(balance_score),
            detail=balance_detail,
        )
    )

    # ---- Autolabel trust ----
    has_autolabel = any(a.confidence is not None for a in dataset.annotations)
    if has_autolabel:
        autolabel_penalty = 0.0
        reasons: List[str] = []
        for f in by_check.get("confidence_low_pass", []):
            # Aggregated per class; penalty scales with affected-count
            # fraction across the whole dataset.
            count = int(f.evidence.get("count", 0))
            autolabel_penalty += min(30.0, 30.0 * count / max(1, total_ann))
            reasons.append(
                f"low-conf {f.evidence.get('class_name', '?')} ({count})"
            )
        for f in by_check.get("confidence_size_mismatch", []):
            autolabel_penalty += 5.0  # each overconfident-tiny is notable
        if by_check.get("confidence_class_bias"):
            autolabel_penalty += 10.0
            reasons.append("per-class confidence spread")
        autolabel_score = max(0.0, 100.0 - autolabel_penalty)
        autolabel_detail = (
            "Autolabels look clean."
            if not reasons
            else "; ".join(reasons[:3])
        )
    else:
        autolabel_score = -1.0
        autolabel_detail = "No autolabels (confidence) in this dataset."
    subscores.append(
        Subscore(
            name="Autolabel trust",
            key="autolabel",
            score=autolabel_score,
            grade=_letter_grade(autolabel_score) if autolabel_score >= 0 else "—",
            detail=autolabel_detail,
        )
    )

    # ---- Overall (weighted over computable subscores only) ----
    usable = [s for s in subscores if s.score >= 0]
    if not usable:
        overall: Optional[float] = None
        overall_grade = "—"
    else:
        total_w = sum(SUBSCORE_WEIGHTS[s.key] for s in usable)
        overall = sum(SUBSCORE_WEIGHTS[s.key] * s.score for s in usable) / total_w
        overall_grade = _letter_grade(overall)

    # ---- Top-5 issues (surface directly in the scorecard) ----
    issues = [
        f.title
        for f in sorted(findings, key=lambda f: f.severity)[:5]
    ]

    return HealthScore(
        overall=overall,
        grade=overall_grade,
        subscores=subscores,
        issues=issues,
    )
