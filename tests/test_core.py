"""Tests for the core/ refactor (step 1 of the check-based architecture)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lightly_insights import checks  # noqa: F401 -- registers checks
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
    Severity,
    list_checks,
    run_all,
)
from lightly_insights.core.geometry import Box, Mask, Polygon


# ---- geometry primitives ----

def test_box_area_and_iou() -> None:
    a = Box(0, 0, 10, 10)
    b = Box(5, 0, 15, 10)
    assert a.area == 100
    assert a.aspect_ratio == 1.0
    # Half overlap -> IoU = 50 / (100 + 100 - 50) = 1/3
    assert abs(a.iou(b) - 1 / 3) < 1e-9


def test_polygon_area_tight_box() -> None:
    triangle = Polygon(points=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    assert triangle.area == 50.0
    tb = triangle.tight_box
    assert (tb.xmin, tb.ymin, tb.xmax, tb.ymax) == (0.0, 0.0, 10.0, 10.0)


def test_mask_area_and_components() -> None:
    arr = np.zeros((10, 10), dtype=bool)
    arr[0:3, 0:3] = True   # one 3x3 component
    arr[7:10, 7:10] = True  # a second, disconnected component
    mask = Mask(array=arr)
    assert mask.area == 18
    assert mask.connected_components() == 2
    tb = mask.tight_box
    assert (tb.xmin, tb.ymin, tb.xmax, tb.ymax) == (0.0, 0.0, 10.0, 10.0)


# ---- Dataset derived views ----

def _make_dataset(
    boxes_per_image: dict,
    categories: list,
) -> Dataset:
    cats = [Category(id=cid, name=name) for cid, name in categories]
    images = [
        Image(filename=f, width=500, height=500) for f in boxes_per_image
    ]
    ann_id = 0
    anns = []
    for fname, boxes in boxes_per_image.items():
        for cid, (xmin, ymin, xmax, ymax) in boxes:
            anns.append(
                Annotation(
                    annotation_id=ann_id,
                    image_filename=fname,
                    class_id=cid,
                    class_name=next(c[1] for c in categories if c[0] == cid),
                    kind=AnnotationKind.BOX,
                    geometry=Box(xmin, ymin, xmax, ymax),
                )
            )
            ann_id += 1
    return Dataset(images=images, annotations=anns, categories=cats)


def test_dataset_derived_views() -> None:
    ds = _make_dataset(
        boxes_per_image={
            "a.jpg": [(0, (0, 0, 10, 10)), (1, (20, 20, 30, 30))],
            "b.jpg": [(0, (5, 5, 15, 15))],
        },
        categories=[(0, "car"), (1, "truck")],
    )
    assert ds.num_images == 2
    assert ds.num_annotations == 3
    assert ds.num_classes == 2
    assert set(ds.annotations_by_image) == {"a.jpg", "b.jpg"}
    assert len(ds.annotations_by_class[0]) == 2
    assert ds.kinds == frozenset({AnnotationKind.BOX})


# ---- Check contract ----

def test_registry_lists_bundled_checks() -> None:
    registered = list_checks()
    for expected in ("corrupt_images", "starved_class", "class_conflict"):
        assert expected in registered


def test_corrupt_images_check_emits_finding() -> None:
    ds = Dataset(
        images=[Image(filename="ok.jpg", width=100, height=100)],
        annotations=[],
        categories=[],
        corrupt_filenames=["broken.png", "weird.tif"],
    )
    findings = run_all(ds, only=["corrupt_images"])
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == Severity.CRITICAL
    assert set(f.affected_images) == {"broken.png", "weird.tif"}


def test_starved_class_check_flags_low_counts() -> None:
    ds = _make_dataset(
        boxes_per_image={
            "a.jpg": [(0, (0, 0, 10, 10))] * 50,  # common
            "b.jpg": [(1, (0, 0, 10, 10))] * 3,   # starved
        },
        categories=[(0, "common"), (1, "rare")],
    )
    findings = run_all(ds, only=["starved_class"])
    assert len(findings) == 1
    assert "rare" in findings[0].title


def test_starved_class_check_flags_orphan_class() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[],
        categories=[Category(id=0, name="orphan")],
    )
    findings = run_all(ds, only=["starved_class"])
    assert len(findings) == 1
    assert "no annotations" in findings[0].title


def test_class_conflict_check_ignores_same_class() -> None:
    ds = _make_dataset(
        boxes_per_image={
            "a.jpg": [
                (0, (10, 10, 100, 100)),
                (0, (12, 12, 102, 102)),  # same class -> not a conflict
                (1, (14, 14, 104, 104)),  # different class, overlapping
            ],
        },
        categories=[(0, "car"), (1, "truck")],
    )
    findings = run_all(ds, only=["class_conflict"])
    assert len(findings) == 2  # car↔truck + car↔truck (the pairs involving class 1)
    for f in findings:
        evidence = f.evidence
        assert evidence["class_a"] != evidence["class_b"]


def test_run_all_sorted_by_severity() -> None:
    # Two starved classes + one conflict. Starved is HIGH (30), conflict HIGH (30),
    # corrupt is CRITICAL (10) -> corrupt first.
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            Annotation(
                annotation_id=0,
                image_filename="a.jpg",
                class_id=0,
                class_name="car",
                kind=AnnotationKind.BOX,
                geometry=Box(10, 10, 50, 50),
            ),
            Annotation(
                annotation_id=1,
                image_filename="a.jpg",
                class_id=1,
                class_name="truck",
                kind=AnnotationKind.BOX,
                geometry=Box(12, 12, 52, 52),
            ),
        ],
        categories=[Category(id=0, name="car"), Category(id=1, name="truck")],
        corrupt_filenames=["bad.jpg"],
    )
    findings = run_all(ds)
    # First finding must be the CRITICAL one.
    assert findings[0].check_id == "corrupt_images"
    assert findings[0].severity == Severity.CRITICAL
    # All others are >= HIGH
    for f in findings[1:]:
        assert f.severity >= Severity.HIGH
