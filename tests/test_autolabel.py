"""Tests for autolabel-aware data model + checks + review queue."""
from __future__ import annotations

from pathlib import Path

from lightly_insights import checks  # noqa: F401
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Finding,
    Image,
    Severity,
    build_review_queue,
    export_review_queue_csv,
    list_checks,
    run_all,
)
from lightly_insights.core.geometry import Box


def _ann(
    idx: int,
    filename: str,
    class_id: int,
    class_name: str,
    box: Box,
    confidence: "float | None" = None,
    source: "str | None" = None,
) -> Annotation:
    return Annotation(
        annotation_id=idx,
        image_filename=filename,
        class_id=class_id,
        class_name=class_name,
        kind=AnnotationKind.BOX,
        geometry=box,
        confidence=confidence,
        source=source,
    )


# ---- Annotation schema ----

def test_annotation_defaults_confidence_none() -> None:
    ann = _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10))
    assert ann.confidence is None
    assert ann.source is None
    assert ann.is_autolabeled is False


def test_annotation_autolabel_when_source_set() -> None:
    ann = _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.8, source="yolo")
    assert ann.is_autolabeled is True


def test_annotation_human_labels_not_autolabeled() -> None:
    ann = _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), source="human")
    assert ann.is_autolabeled is False


# ---- Registry ----

def test_autolabel_checks_registered() -> None:
    registry = list_checks()
    for cid in (
        "confidence_low_pass",
        "confidence_size_mismatch",
        "confidence_class_bias",
    ):
        assert cid in registry


# ---- Confidence low-pass ----

def test_confidence_low_pass_skips_datasets_without_confidence() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10))],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["confidence_low_pass"])
    assert findings == []


def test_confidence_low_pass_flags_below_threshold() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.15, source="yolo"),
            _ann(1, "a.jpg", 0, "car", Box(20, 20, 30, 30), confidence=0.85, source="yolo"),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["confidence_low_pass"])
    # Aggregated per class: one finding even with N affected annotations.
    assert len(findings) == 1
    assert findings[0].affected_annotations == [0]
    assert findings[0].evidence["count"] == 1
    assert findings[0].evidence["class_name"] == "car"


def test_confidence_low_pass_aggregates_per_class() -> None:
    """500 low-confidence annotations should yield 1 finding, not 500."""
    anns = []
    for i in range(500):
        anns.append(_ann(
            i, f"i_{i}.jpg", 0, "car", Box(0, 0, 10, 10),
            confidence=0.1, source="yolo",
        ))
    imgs = [Image(filename=f"i_{i}.jpg", width=100, height=100) for i in range(500)]
    ds = Dataset(
        images=imgs, annotations=anns,
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["confidence_low_pass"])
    assert len(findings) == 1
    assert findings[0].evidence["count"] == 500
    assert len(findings[0].affected_annotations) == 500


# ---- Confidence × size mismatch ----

def test_confidence_size_mismatch_flags_overconfident_tiny() -> None:
    # 10x10 on 1000x1000 = 0.01 % area < 0.5 %, confidence 0.97
    ds = Dataset(
        images=[Image(filename="a.jpg", width=1000, height=1000)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.97, source="yolo"),
            _ann(1, "a.jpg", 0, "car", Box(0, 0, 300, 300), confidence=0.97, source="yolo"),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["confidence_size_mismatch"])
    assert len(findings) == 1
    assert findings[0].affected_annotations == [0]


def test_confidence_size_mismatch_skips_low_confidence() -> None:
    # Same tiny box, low confidence — handled by low_pass instead.
    ds = Dataset(
        images=[Image(filename="a.jpg", width=1000, height=1000)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.5, source="yolo"),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["confidence_size_mismatch"])
    assert findings == []


# ---- Confidence class bias ----

def test_confidence_class_bias_flags_big_spread() -> None:
    anns = []
    for i in range(50):
        anns.append(_ann(
            i, f"a_{i}.jpg", 0, "car", Box(0, 0, 10, 10),
            confidence=0.95, source="yolo",
        ))
        anns.append(_ann(
            100 + i, f"b_{i}.jpg", 1, "bike", Box(0, 0, 10, 10),
            confidence=0.50, source="yolo",
        ))
    imgs = [Image(filename=f"a_{i}.jpg", width=100, height=100) for i in range(50)]
    imgs += [Image(filename=f"b_{i}.jpg", width=100, height=100) for i in range(50)]
    ds = Dataset(
        images=imgs,
        annotations=anns,
        categories=[Category(id=0, name="car"), Category(id=1, name="bike")],
    )
    findings = run_all(ds, only=["confidence_class_bias"])
    assert len(findings) == 1
    ev = findings[0].evidence
    assert ev["best_class"] == "car"
    assert ev["worst_class"] == "bike"
    assert ev["spread"] > 0.4


def test_confidence_class_bias_passes_on_uniform() -> None:
    anns = []
    for i in range(50):
        for cid, name in ((0, "car"), (1, "bike")):
            anns.append(_ann(
                len(anns), f"x_{cid}_{i}.jpg", cid, name, Box(0, 0, 10, 10),
                confidence=0.80, source="yolo",
            ))
    imgs = list({Image(filename=a.image_filename, width=100, height=100) for a in anns})
    ds = Dataset(
        images=imgs,
        annotations=anns,
        categories=[Category(id=0, name="car"), Category(id=1, name="bike")],
    )
    findings = run_all(ds, only=["confidence_class_bias"])
    assert findings == []


# ---- Review queue ----

def test_review_queue_prioritizes_high_severity_low_confidence(tmp_path: Path) -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.2, source="yolo"),
            _ann(1, "a.jpg", 0, "car", Box(20, 20, 30, 30), confidence=0.95, source="yolo"),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = [
        # CRITICAL severity, annotation 0 (low confidence) -> top priority.
        Finding(
            check_id="critical_bad",
            severity=Severity.CRITICAL,
            category="quality",
            title="Critical",
            detail="...",
            action="review",
            affected_annotations=[0],
            affected_images=["a.jpg"],
        ),
        # MEDIUM severity, annotation 1 (high confidence) -> lower.
        Finding(
            check_id="minor",
            severity=Severity.MEDIUM,
            category="quality",
            title="Medium",
            detail="...",
            action="review",
            affected_annotations=[1],
            affected_images=["a.jpg"],
        ),
    ]
    queue = build_review_queue(findings=findings, dataset=ds)
    assert len(queue) == 2
    assert queue[0].annotation_id == 0  # low confidence + critical
    assert queue[0].rank == 1
    assert queue[1].annotation_id == 1
    # Higher priority means larger numeric score in our convention.
    assert queue[0].priority > queue[1].priority

    # CSV export smoke test.
    out = tmp_path / "queue.csv"
    export_review_queue_csv(queue, out)
    lines = out.read_text().splitlines()
    assert lines[0].startswith("rank,priority,finding")
    assert len(lines) == 3  # header + 2 items


def test_review_queue_handles_dataset_level_findings() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[],
        categories=[],
    )
    findings = [
        Finding(
            check_id="dataset_level",
            severity=Severity.HIGH,
            category="balance",
            title="Dataset-level",
            detail="...",
            action="fix something",
        )
    ]
    queue = build_review_queue(findings=findings, dataset=ds)
    assert len(queue) == 1
    assert queue[0].annotation_id is None
    assert queue[0].filename == ""


def test_review_queue_max_items_truncates() -> None:
    ds = Dataset(
        images=[Image(filename=f"i_{i}.jpg", width=100, height=100) for i in range(10)],
        annotations=[
            _ann(i, f"i_{i}.jpg", 0, "car", Box(0, 0, 10, 10), confidence=0.1 * i)
            for i in range(10)
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = [
        Finding(
            check_id="x",
            severity=Severity.MEDIUM,
            category="quality",
            title="t",
            detail="d",
            action="a",
            affected_annotations=[i],
            affected_images=[f"i_{i}.jpg"],
        )
        for i in range(10)
    ]
    queue = build_review_queue(findings=findings, dataset=ds, max_items=3)
    assert len(queue) == 3
    # Lowest-confidence item should rank first.
    assert queue[0].annotation_id == 0
