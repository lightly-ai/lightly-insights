"""Tests for pretrained-model proposal + missing-label detection."""
from __future__ import annotations

import pytest

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
from lightly_insights.ml import PROPOSAL_SOURCE_PREFIX, is_proposal_source


def _ann(idx, filename, cid, name, box, source, confidence=None):
    return Annotation(
        annotation_id=idx,
        image_filename=filename,
        class_id=cid,
        class_name=name,
        kind=AnnotationKind.BOX,
        geometry=box,
        confidence=confidence,
        source=source,
    )


def test_is_proposal_source_helper() -> None:
    assert is_proposal_source("proposal:yolov8n")
    assert is_proposal_source("proposal:sam")
    assert not is_proposal_source("human")
    assert not is_proposal_source("yolo-v11")
    assert not is_proposal_source(None)


def test_check_registered() -> None:
    assert "missing_label_proposal" in list_checks()


def test_skipped_without_proposals() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[_ann(0, "a.jpg", 0, "car", Box(0, 0, 10, 10), "human")],
        categories=[Category(id=0, name="car")],
    )
    assert run_all(ds, only=["missing_label_proposal"]) == []


def test_flags_unmatched_high_confidence_proposal() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "human"),
            # Proposal in a completely different region → unmatched.
            _ann(
                10_000_000, "a.jpg", 0, "car", Box(300, 300, 400, 400),
                f"{PROPOSAL_SOURCE_PREFIX}yolov8n", confidence=0.85,
            ),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["missing_label_proposal"])
    assert len(findings) == 1
    assert findings[0].affected_annotations == [10_000_000]
    assert "candidate missing label" in findings[0].title.lower()


def test_skips_matched_proposal() -> None:
    # Proposal overlaps human label → not missing.
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "human"),
            _ann(
                10_000_000, "a.jpg", 0, "car", Box(12, 12, 102, 102),
                f"{PROPOSAL_SOURCE_PREFIX}yolov8n", confidence=0.85,
            ),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["missing_label_proposal"])
    assert findings == []


def test_skips_low_confidence_proposal() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[
            _ann(0, "a.jpg", 0, "car", Box(10, 10, 100, 100), "human"),
            # Unmatched but low-confidence → skip (noise filter).
            _ann(
                10_000_000, "a.jpg", 0, "car", Box(300, 300, 400, 400),
                f"{PROPOSAL_SOURCE_PREFIX}yolov8n", confidence=0.3,
            ),
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["missing_label_proposal"])
    assert findings == []


def test_propose_with_yolo_import_error_message() -> None:
    """When ultralytics is not installed, the error message points to the [ml] extra."""
    import importlib.util
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics installed; can't test ImportError path")
    from lightly_insights.ml import propose_with_yolo
    with pytest.raises(ImportError) as exc_info:
        propose_with_yolo(image_paths=[])
    assert "lightly_insights[ml]" in str(exc_info.value)
