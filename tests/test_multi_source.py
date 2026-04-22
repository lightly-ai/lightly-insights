"""Tests for multi-source disagreement check."""
from __future__ import annotations

from lightly_insights import checks  # noqa: F401
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
    list_checks,
    run_all,
)
from lightly_insights.core.geometry import Box


def _ann(idx, filename, cid, name, box, source):
    return Annotation(
        annotation_id=idx, image_filename=filename, class_id=cid,
        class_name=name, kind=AnnotationKind.BOX, geometry=box, source=source,
    )


def test_check_registered() -> None:
    assert "multi_source_disagreement" in list_checks()


def test_skipped_with_single_source() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), "yolo")],
        categories=[Category(id=0, name="car")],
    )
    assert run_all(ds, only=["multi_source_disagreement"]) == []


def test_detects_class_disagreement() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=200, height=200)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "yolo"),
            _ann(1, "a.jpg", 1, "truck", Box(12, 12, 102, 102), "sam"),
        ],
        categories=[Category(id=0, name="car"), Category(id=1, name="truck")],
    )
    findings = run_all(ds, only=["multi_source_disagreement"])
    disagreement = [f for f in findings if "disagree on class" in f.title]
    assert len(disagreement) == 1


def test_detects_unmatched_in_each_source() -> None:
    # yolo has a car at (10,10)-(100,100); sam has nothing. yolo's annotation
    # is unmatched → MEDIUM finding. No reference source → symmetric.
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "yolo"),
            _ann(1, "a.jpg", 0, "car", Box(300, 300, 400, 400), "sam"),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["multi_source_disagreement"])
    # Both unmatched.
    assert len(findings) == 2
    for f in findings:
        assert "has a 'car' that" in f.title or "missed" in f.title


def test_human_source_is_reference() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "human"),
            _ann(1, "a.jpg", 0, "car", Box(300, 300, 400, 400), "yolo"),  # different location
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["multi_source_disagreement"])
    # With human as reference:
    # - human's car has no yolo match → "yolo missed ..." (HIGH severity)
    # - yolo's car has no human match → "yolo has a 'car' that human doesn't" (MEDIUM)
    assert len(findings) == 2
    severities = sorted(f.severity for f in findings)
    # Expect one HIGH (30) and one MEDIUM (50).
    assert 30 in severities
    assert 50 in severities
