"""Tests for the geometry-aware checks added in step 3."""
from __future__ import annotations

import numpy as np

from lightly_insights import checks  # noqa: F401 -- registers checks
from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
    list_checks,
    run_all,
)
from lightly_insights.core.geometry import Box, Mask, Polygon


def _ann(
    idx: int,
    filename: str,
    class_id: int,
    class_name: str,
    kind: AnnotationKind,
    geometry: object,
) -> Annotation:
    return Annotation(
        annotation_id=idx,
        image_filename=filename,
        class_id=class_id,
        class_name=class_name,
        kind=kind,
        geometry=geometry,  # type: ignore[arg-type]
    )


# ---- Registry ----

def test_all_step3_checks_registered() -> None:
    for cid in (
        "shape_outlier",
        "polygon_self_intersect",
        "polygon_axis_aligned",
        "mask_fragmentation",
        "duplicate_annotation",
    ):
        assert cid in list_checks()


# ---- Shape outlier ----

def test_shape_outlier_flags_giant_box_in_class_of_small_boxes() -> None:
    # 25 small boxes + 1 giant.
    small = [Box(0, 0, 20, 20) for _ in range(25)]
    giant = Box(0, 0, 500, 500)
    anns = []
    for i, b in enumerate(small):
        anns.append(_ann(i, f"img_{i}.jpg", 0, "car", AnnotationKind.BOX, b))
    anns.append(_ann(25, "outlier.jpg", 0, "car", AnnotationKind.BOX, giant))

    ds = Dataset(
        images=[Image(filename=f"img_{i}.jpg", width=1000, height=1000) for i in range(25)]
        + [Image(filename="outlier.jpg", width=1000, height=1000)],
        annotations=anns,
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["shape_outlier"])
    assert len(findings) == 1
    assert findings[0].affected_images == ["outlier.jpg"]
    assert findings[0].evidence["class_name"] == "car"


def test_shape_outlier_skips_classes_with_too_few_samples() -> None:
    # Only 5 annotations — below MIN_CLASS_SAMPLES=20; should not flag anything.
    anns = [
        _ann(i, f"img_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10 ** (i + 1), 10))
        for i in range(5)
    ]
    ds = Dataset(
        images=[Image(filename=f"img_{i}.jpg", width=10000, height=1000) for i in range(5)],
        annotations=anns,
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["shape_outlier"])
    assert findings == []


# ---- Polygon self-intersect ----

def test_polygon_self_intersect_detects_figure_eight() -> None:
    # Classic bowtie quadrilateral: edges cross.
    bowtie = Polygon(points=((0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)))
    ann = _ann(0, "bowtie.jpg", 0, "shape", AnnotationKind.POLYGON, bowtie)
    ds = Dataset(
        images=[Image(filename="bowtie.jpg", width=100, height=100)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["polygon_self_intersect"])
    assert len(findings) == 1


def test_polygon_self_intersect_skips_simple_polygon() -> None:
    square = Polygon(points=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
    ann = _ann(0, "square.jpg", 0, "shape", AnnotationKind.POLYGON, square)
    ds = Dataset(
        images=[Image(filename="square.jpg", width=100, height=100)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["polygon_self_intersect"])
    assert findings == []


# ---- Polygon axis-aligned ----

def test_polygon_axis_aligned_flags_rectangle_trace() -> None:
    # 8-vertex shape, all edges horizontal or vertical.
    trace = Polygon(points=(
        (0.0, 0.0), (50.0, 0.0), (50.0, 20.0), (80.0, 20.0),
        (80.0, 50.0), (50.0, 50.0), (50.0, 80.0), (0.0, 80.0),
    ))
    ann = _ann(0, "trace.jpg", 0, "shape", AnnotationKind.POLYGON, trace)
    ds = Dataset(
        images=[Image(filename="trace.jpg", width=100, height=100)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["polygon_axis_aligned"])
    assert len(findings) == 1
    assert findings[0].evidence["axis_aligned_fraction"] == 1.0


def test_polygon_axis_aligned_skips_organic_shape() -> None:
    # 12-vertex star with diagonal edges.
    import math
    points = []
    for i in range(12):
        r = 10 if i % 2 == 0 else 5
        angle = (2 * math.pi * i) / 12
        points.append((r * math.cos(angle), r * math.sin(angle)))
    star = Polygon(points=tuple(points))
    ann = _ann(0, "star.jpg", 0, "shape", AnnotationKind.POLYGON, star)
    ds = Dataset(
        images=[Image(filename="star.jpg", width=100, height=100)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["polygon_axis_aligned"])
    assert findings == []


# ---- Mask fragmentation ----

def test_mask_fragmentation_flags_many_blobs() -> None:
    arr = np.zeros((50, 50), dtype=bool)
    # Five separate 5x5 blobs.
    for r in (0, 15, 30):
        for c in (0, 15):
            arr[r:r + 5, c:c + 5] = True
    # One more at 40,30 to make 7 total
    arr[40:45, 30:35] = True
    ann = _ann(
        0,
        "frag.jpg",
        0,
        "shape",
        AnnotationKind.MASK,
        Mask(array=arr),
    )
    ds = Dataset(
        images=[Image(filename="frag.jpg", width=50, height=50)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["mask_fragmentation"])
    assert len(findings) == 1
    assert findings[0].evidence["significant_components"] >= 5


def test_mask_fragmentation_skips_single_blob() -> None:
    arr = np.zeros((50, 50), dtype=bool)
    arr[10:40, 10:40] = True
    ann = _ann(
        0,
        "solid.jpg",
        0,
        "shape",
        AnnotationKind.MASK,
        Mask(array=arr),
    )
    ds = Dataset(
        images=[Image(filename="solid.jpg", width=50, height=50)],
        annotations=[ann],
        categories=[Category(id=0, name="shape")],
    )
    findings = run_all(ds, only=["mask_fragmentation"])
    assert findings == []


# ---- Duplicate annotation (cross-geometry port) ----

def test_duplicate_annotation_flags_near_identical_boxes() -> None:
    a = _ann(0, "img.jpg", 0, "car", AnnotationKind.BOX, Box(10, 10, 100, 100))
    b = _ann(1, "img.jpg", 0, "car", AnnotationKind.BOX, Box(11, 11, 101, 101))
    ds = Dataset(
        images=[Image(filename="img.jpg", width=500, height=500)],
        annotations=[a, b],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["duplicate_annotation"])
    assert len(findings) == 1


def test_duplicate_annotation_ignores_different_class() -> None:
    a = _ann(0, "img.jpg", 0, "car", AnnotationKind.BOX, Box(10, 10, 100, 100))
    b = _ann(1, "img.jpg", 1, "truck", AnnotationKind.BOX, Box(11, 11, 101, 101))
    ds = Dataset(
        images=[Image(filename="img.jpg", width=500, height=500)],
        annotations=[a, b],
        categories=[
            Category(id=0, name="car"),
            Category(id=1, name="truck"),
        ],
    )
    findings = run_all(ds, only=["duplicate_annotation"])
    assert findings == []


def test_duplicate_annotation_works_for_polygons() -> None:
    # Polygons with near-identical tight bounding boxes.
    poly_a = Polygon(points=((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)))
    poly_b = Polygon(points=((11.0, 11.0), (51.0, 11.0), (51.0, 51.0), (11.0, 51.0)))
    a = _ann(0, "img.jpg", 0, "leaf", AnnotationKind.POLYGON, poly_a)
    b = _ann(1, "img.jpg", 0, "leaf", AnnotationKind.POLYGON, poly_b)
    ds = Dataset(
        images=[Image(filename="img.jpg", width=200, height=200)],
        annotations=[a, b],
        categories=[Category(id=0, name="leaf")],
    )
    findings = run_all(ds, only=["duplicate_annotation"])
    assert len(findings) == 1
