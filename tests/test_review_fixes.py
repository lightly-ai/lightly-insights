"""Tests for the fixes from the correctness review."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lightly_insights import checks  # noqa: F401 -- register checks
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Finding,
    Image,
    Severity,
    build_review_queue,
    register_check,
    run_all,
)
from lightly_insights.core.check import Check
from lightly_insights.core.geometry import Box


# ---- A5: sort tiebreak handles annotation_id=0 correctly ----

def test_review_queue_tiebreak_preserves_annotation_id_zero() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            Annotation(
                annotation_id=0, image_filename="a.jpg", class_id=0,
                class_name="x", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            ),
            Annotation(
                annotation_id=5, image_filename="a.jpg", class_id=0,
                class_name="x", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            ),
        ],
        categories=[Category(id=0, name="x")],
    )
    # Same severity, same filename → stable by annotation_id ascending.
    findings = [
        Finding(
            check_id="x", severity=Severity.MEDIUM, category="a",
            title="t", detail="d", action="a",
            affected_annotations=[5], affected_images=["a.jpg"],
        ),
        Finding(
            check_id="x", severity=Severity.MEDIUM, category="a",
            title="t", detail="d", action="a",
            affected_annotations=[0], affected_images=["a.jpg"],
        ),
    ]
    queue = build_review_queue(findings, ds)
    assert len(queue) == 2
    # annotation_id 0 should come first (not sorted as None).
    assert queue[0].annotation_id == 0
    assert queue[1].annotation_id == 5


def test_review_queue_sorts_none_annotation_ids_last() -> None:
    """When everything else ties, annotation_id=0 wins over annotation_id=None.

    Isolates the ``annotation_id or 0`` sort bug where None collapsed to 0,
    making the two rows indistinguishable.
    """
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            Annotation(
                annotation_id=0, image_filename="a.jpg", class_id=0,
                class_name="x", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            ),
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = [
        Finding(
            check_id="x", severity=Severity.MEDIUM, category="a",
            title="with_ann", detail="d", action="a",
            affected_annotations=[0], affected_images=["a.jpg"],
        ),
        Finding(  # image-level, no specific ann; shares the same filename
            check_id="x", severity=Severity.MEDIUM, category="a",
            title="image_level", detail="d", action="a",
            affected_images=["a.jpg"],
        ),
    ]
    queue = build_review_queue(findings, ds)
    # Same priority, severity, filename → tiebreak on annotation_id.
    # None should sort AFTER 0 (not be conflated via ``or 0``).
    assert len(queue) == 2
    assert queue[0].annotation_id == 0
    assert queue[1].annotation_id is None


# ---- A11: severity buckets cover the whole 0-100 range ----

def test_severity_buckets_cover_all_custom_severities() -> None:
    from lightly_insights.findings_present import _severity_buckets

    findings = [
        Finding(
            check_id="x", severity=sev, category="a",
            title="t", detail="d", action="a",
        )
        for sev in (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)
    ]
    buckets = _severity_buckets(findings)
    total = sum(count for _, count, _ in buckets)
    # Every finding must be accounted for — no gaps.
    assert total == len(findings)


# ---- A9: confidence_low_pass aggregates by class ----

def test_confidence_low_pass_produces_one_finding_per_class_at_scale() -> None:
    anns = []
    for i in range(300):
        anns.append(Annotation(
            annotation_id=i, image_filename=f"i_{i}.jpg", class_id=i % 3,
            class_name=["car", "truck", "bike"][i % 3],
            kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            confidence=0.1, source="yolo",
        ))
    ds = Dataset(
        images=[Image(filename=f"i_{i}.jpg", width=100, height=100) for i in range(300)],
        annotations=anns,
        categories=[
            Category(id=0, name="car"),
            Category(id=1, name="truck"),
            Category(id=2, name="bike"),
        ],
    )
    findings = run_all(ds, only=["confidence_low_pass"])
    # Exactly 3 findings — one per class — not 300.
    assert len(findings) == 3
    # Each carries 100 affected annotations.
    for f in findings:
        assert len(f.affected_annotations) == 100


# ---- R15: run_all isolates crashing checks ----

def test_run_all_isolates_crashing_check(monkeypatch) -> None:
    @register_check
    class _CrashingCheck(Check):
        check_id = "_crashing_check_for_test"
        title = "crash"
        category = "meta"

        def run(self, dataset):
            raise ValueError("intentional")

    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[],
        categories=[],
    )
    # Non-strict: crash is caught, a meta-finding is emitted.
    findings = run_all(ds, only=["_crashing_check_for_test"])
    assert len(findings) == 1
    assert findings[0].check_id == "_crashing_check_for_test"
    assert findings[0].category == "meta"
    assert "intentional" in findings[0].evidence["exception"]

    # Strict: the original exception should bubble up.
    with pytest.raises(ValueError, match="intentional"):
        run_all(ds, only=["_crashing_check_for_test"], strict=True)


# ---- R11: warn on duplicate check_id registration ----

def test_register_check_warns_on_duplicate(caplog) -> None:
    class _A(Check):
        check_id = "_dup_test_check"
        title = "a"
        category = "meta"

        def run(self, dataset):
            return []

    class _B(Check):
        check_id = "_dup_test_check"  # same id
        title = "b"
        category = "meta"

        def run(self, dataset):
            return []

    register_check(_A)
    with caplog.at_level(logging.WARNING, logger="lightly_insights.core.check"):
        register_check(_B)
    assert any("being overwritten" in m for m in caplog.messages)


# ---- B5: adapter only sets path= for files that exist ----

def test_adapter_leaves_path_none_for_missing_files(tmp_path: Path) -> None:
    from lightly_insights.analyze import ImageAnalysis
    from collections import Counter
    from lightly_insights.core.adapter import build_dataset

    # Folder with one real file + one manifest entry without file on disk.
    (tmp_path / "real.png").write_bytes(b"")  # empty but exists
    analysis = ImageAnalysis(
        num_images=2,
        image_folder=tmp_path,
        filename_set={"real.png", "missing.png"},
        image_sizes=Counter(),
        median_size=(0, 0),
        corrupt_files=[],
        filename_to_size={"real.png": (10, 10), "missing.png": (20, 20)},
    )
    ds = build_dataset(image_analysis=analysis)
    by_name = {img.filename: img for img in ds.images}
    assert by_name["real.png"].path is not None
    assert by_name["missing.png"].path is None
