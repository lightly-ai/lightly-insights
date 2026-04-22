"""Tests for the findings-aware health scorecard."""
from __future__ import annotations

from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Finding,
    Image,
    Severity,
    compute_health,
)
from lightly_insights.core.geometry import Box


def _ann(idx, cid, name, conf=None, source=None):
    return Annotation(
        annotation_id=idx,
        image_filename="a.jpg",
        class_id=cid,
        class_name=name,
        kind=AnnotationKind.BOX,
        geometry=Box(0, 0, 10, 10),
        confidence=conf,
        source=source,
    )


def _f(check_id, severity, category="annotation", affected_ann=None, **evidence):
    return Finding(
        check_id=check_id,
        severity=severity,
        category=category,
        title=f"{check_id} title",
        detail="d",
        action="a",
        affected_annotations=affected_ann or [],
        affected_images=["a.jpg"] if affected_ann else [],
        evidence=evidence,
    )


def test_clean_dataset_scores_a() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(i, 0, "car") for i in range(100)],
        categories=[Category(id=0, name="car")],
    )
    score = compute_health(ds, [])
    assert score.overall is not None and score.overall >= 95
    assert score.grade == "A"
    # Only pipeline + annotation + balance subscores are computable when no
    # confidence is present (autolabel trust is dropped).
    computable = [s for s in score.subscores if s.score >= 0]
    assert [s.key for s in computable] == ["pipeline", "annotation", "balance"]


def test_critical_finding_tanks_pipeline_subscore() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, 0, "car")],
        categories=[Category(id=0, name="car")],
        corrupt_filenames=["broken.png"],
    )
    findings = [_f("corrupt_images", Severity.CRITICAL, category="quality")]
    score = compute_health(ds, findings)
    pipeline = next(s for s in score.subscores if s.key == "pipeline")
    assert pipeline.score == 70  # -30 for one critical


def test_annotation_subscore_reflects_flagged_ratio() -> None:
    # 50 annotations, 25 flagged by annotation-category findings.
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(i, 0, "car") for i in range(50)],
        categories=[Category(id=0, name="car")],
    )
    findings = [
        _f("class_conflict", Severity.HIGH, affected_ann=list(range(25))),
    ]
    score = compute_health(ds, findings)
    ann = next(s for s in score.subscores if s.key == "annotation")
    assert abs(ann.score - 50.0) < 1e-6


def test_balance_subscore_penalizes_starved_and_leakage() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, 0, "rare")],
        categories=[Category(id=0, name="rare")],
    )
    findings = [
        _f("starved_class", Severity.HIGH, category="balance",
           affected_ann=[], class_name="rare"),
        _f("split_leakage", Severity.CRITICAL, category="balance"),
    ]
    score = compute_health(ds, findings)
    balance = next(s for s in score.subscores if s.key == "balance")
    # -15 (starved) - 30 (leakage) = -45 → 55
    assert balance.score == 55.0


def test_autolabel_subscore_dropped_without_confidences() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, 0, "car")],
        categories=[Category(id=0, name="car")],
    )
    score = compute_health(ds, [])
    autolabel = next(s for s in score.subscores if s.key == "autolabel")
    assert autolabel.score == -1  # N/A
    assert autolabel.grade == "—"


def test_autolabel_subscore_active_with_confidences() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(i, 0, "car", conf=0.95 if i < 90 else 0.1, source="yolo")
            for i in range(100)
        ],
        categories=[Category(id=0, name="car")],
    )
    # Simulate confidence_low_pass emitting one aggregated finding.
    findings = [
        Finding(
            check_id="confidence_low_pass",
            severity=Severity.MEDIUM,
            category="annotation",
            title="10 low-confidence 'car' annotation(s)",
            detail="...",
            action="review",
            affected_annotations=list(range(90, 100)),
            affected_images=["a.jpg"],
            evidence={"class_name": "car", "count": 10, "mean_confidence": 0.1},
        ),
    ]
    score = compute_health(ds, findings)
    autolabel = next(s for s in score.subscores if s.key == "autolabel")
    assert 0 <= autolabel.score < 100


def test_overall_reweights_when_some_subscores_na() -> None:
    """A dataset with no annotations should still produce an overall from
    pipeline alone — the annotation/balance/autolabel subscores collapse
    to N/A without tanking the number."""
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[],
        categories=[],
    )
    score = compute_health(ds, [])
    assert score.overall is not None
    # Pipeline subscore is 100 (no critical findings), annotation is N/A
    # (no annotations). The weighted average of the computable ones should
    # still be 100.
    assert score.overall == 100.0


def test_top_issues_sorted_by_severity() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, 0, "car")],
        categories=[Category(id=0, name="car")],
    )
    findings = [
        _f("x", Severity.LOW, affected_ann=[0]),
        _f("y", Severity.CRITICAL, category="quality"),
        _f("z", Severity.HIGH, affected_ann=[0]),
    ]
    score = compute_health(ds, findings)
    # Critical first, then high, then low.
    assert "y" in score.issues[0]
    assert "z" in score.issues[1]
