"""Smoke tests for the findings-first HTML report."""
from __future__ import annotations

from pathlib import Path

from lightly_insights import checks  # noqa: F401
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
    build_review_queue,
    run_all,
)
from lightly_insights.core.geometry import Box
from lightly_insights.findings_present import create_findings_report


def _build_small_dataset() -> Dataset:
    images = [Image(filename=f"img_{i}.jpg", width=200, height=200) for i in range(10)]
    anns = [
        Annotation(
            annotation_id=i,
            image_filename=f"img_{i}.jpg",
            class_id=0,
            class_name="car",
            kind=AnnotationKind.BOX,
            geometry=Box(10, 10, 50, 50),
            confidence=0.2 if i < 3 else 0.9,
            source="yolo",
        )
        for i in range(10)
    ]
    return Dataset(
        images=images,
        annotations=anns,
        categories=[Category(id=0, name="car")],
        corrupt_filenames=["bad.png"],
    )


def test_report_renders_index_and_sidecars(tmp_path: Path) -> None:
    ds = _build_small_dataset()
    findings = run_all(ds)
    queue = build_review_queue(findings, ds, max_items=20)

    index = create_findings_report(tmp_path, ds, findings, queue)

    assert index.exists()
    assert index.name == "index.html"
    body = index.read_text()
    # Header + summary strip.
    assert "Dataset Findings" in body
    assert "Review queue" in body
    # Review queue section renders only when there is one.
    if queue:
        assert "Priority review queue" in body
    # Findings-by-category section always present.
    assert "All findings by category" in body

    # Sidecar files.
    assert (tmp_path / "review_queue.csv").exists()
    assert (tmp_path / "findings.json").exists()
    assert (tmp_path / "static").is_dir()


def test_report_survives_empty_findings(tmp_path: Path) -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=10, height=10)],
        annotations=[],
        categories=[],
    )
    index = create_findings_report(tmp_path, ds, findings=[], review_queue=[])
    assert index.exists()
    body = index.read_text()
    assert "Total findings" in body
    # Should not crash even with zero findings / queue.


def test_severity_buckets_split_correctly(tmp_path: Path) -> None:
    from lightly_insights.core import Finding, Severity

    ds = _build_small_dataset()
    findings = [
        Finding(
            check_id="x", severity=Severity.CRITICAL, category="a",
            title="t", detail="d", action="a",
        ),
        Finding(
            check_id="y", severity=Severity.HIGH, category="a",
            title="t", detail="d", action="a",
        ),
        Finding(
            check_id="z", severity=Severity.MEDIUM, category="b",
            title="t", detail="d", action="a",
        ),
    ]
    queue = build_review_queue(findings, ds)
    index = create_findings_report(tmp_path, ds, findings, queue)
    body = index.read_text()
    # Category "a" should come before "b" (lower min severity).
    pos_a = body.find(">A<")  # heading word "A"
    # The slim template renders capitalized category names; just confirm order.
    assert body.find("<summary>\n      A") < body.find("<summary>\n      B")
