"""Tests for the round-1 integrity checks."""
from __future__ import annotations

import numpy as np
import pytest

from lightly_insights import checks  # noqa: F401 -- registers
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


def _ann(idx, filename, class_id, class_name, kind, geometry):
    return Annotation(
        annotation_id=idx, image_filename=filename, class_id=class_id,
        class_name=class_name, kind=kind, geometry=geometry,
    )


# ---- Registry ----

def test_round1_checks_registered() -> None:
    registry = list_checks()
    for cid in (
        "degenerate_annotation",
        "out_of_bounds_annotation",
        "mask_image_size_mismatch",
        "background_scarcity",
        "same_class_overlap",
        "polygon_mask_consistency",
        "split_purity",
        "split_leakage",
    ):
        assert cid in registry, f"{cid} not registered"


# ---- Geometric validity ----

def test_degenerate_box_flagged() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "x", AnnotationKind.BOX, Box(10, 10, 10, 10)),
            _ann(1, "a.jpg", 0, "x", AnnotationKind.BOX, Box(20, 20, 50, 50)),
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = run_all(ds, only=["degenerate_annotation"])
    assert len(findings) == 1
    assert findings[0].affected_annotations == [0]
    assert findings[0].severity == Severity.CRITICAL


def test_degenerate_polygon_flagged() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "x", AnnotationKind.POLYGON,
                 Polygon(points=((0.0, 0.0), (10.0, 10.0)))),  # < 3 points
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = run_all(ds, only=["degenerate_annotation"])
    assert len(findings) == 1
    assert "< 3" in findings[0].detail


def test_out_of_bounds_box_flagged() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "x", AnnotationKind.BOX, Box(-5, 10, 50, 50)),
            _ann(1, "a.jpg", 0, "x", AnnotationKind.BOX, Box(10, 10, 50, 50)),
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = run_all(ds, only=["out_of_bounds_annotation"])
    assert len(findings) == 1
    assert findings[0].affected_annotations == [0]


def test_mask_image_size_mismatch_flagged() -> None:
    arr = np.ones((80, 80), dtype=bool)
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "x", AnnotationKind.MASK, Mask(array=arr)),
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = run_all(ds, only=["mask_image_size_mismatch"])
    assert len(findings) == 1
    assert findings[0].evidence["mask_width"] == 80


# ---- Background scarcity ----

def test_background_scarcity_fires_on_dense_dataset() -> None:
    # 10 images, every one annotated -> 0 % background.
    ds = Dataset(
        images=[Image(filename=f"i_{i}.jpg", width=100, height=100) for i in range(10)],
        annotations=[
            _ann(i, f"i_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10))
            for i in range(10)
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["background_scarcity"])
    assert len(findings) == 1
    assert findings[0].evidence["background_fraction"] == 0.0


def test_background_scarcity_passes_with_sufficient_backgrounds() -> None:
    # 10 images, 2 unannotated -> 20 %. Passes threshold of 5 %.
    ds = Dataset(
        images=[Image(filename=f"i_{i}.jpg", width=100, height=100) for i in range(10)],
        annotations=[
            _ann(i, f"i_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10))
            for i in range(8)
        ],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["background_scarcity"])
    assert findings == []


# ---- Same-class overlap ----

def test_same_class_overlap_flags_enclosed_annotation() -> None:
    # Big car box with a small car box fully inside it.
    big = _ann(0, "a.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 200, 200))
    small = _ann(1, "a.jpg", 0, "car", AnnotationKind.BOX, Box(50, 50, 100, 100))
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[big, small],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["same_class_overlap"])
    assert len(findings) == 1
    assert findings[0].evidence["containment"] == 1.0


def test_same_class_overlap_skips_duplicates() -> None:
    # IoU >= 0.9 -> owned by duplicate_annotation, skipped here.
    a = _ann(0, "a.jpg", 0, "car", AnnotationKind.BOX, Box(10, 10, 100, 100))
    b = _ann(1, "a.jpg", 0, "car", AnnotationKind.BOX, Box(11, 11, 100, 100))
    ds = Dataset(
        images=[Image(filename="a.jpg", width=500, height=500)],
        annotations=[a, b],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["same_class_overlap"])
    assert findings == []


# ---- Polygon/mask consistency ----

def test_polygon_mask_consistency_flags_disagreement() -> None:
    # Triangle polygon vs full-square mask, both sharing the same tight box.
    # Pairing links them (tight-box IoU = 1.0) but rasterized IoU is ~0.5
    # (triangle ≈ half the square's area).
    poly = Polygon(points=((10.0, 10.0), (90.0, 10.0), (50.0, 90.0)))
    mask_arr = np.zeros((100, 100), dtype=bool)
    mask_arr[10:91, 10:91] = True

    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "leaf", AnnotationKind.POLYGON, poly),
            _ann(1, "a.jpg", 0, "leaf", AnnotationKind.MASK, Mask(array=mask_arr)),
        ],
        categories=[Category(id=0, name="leaf")],
    )
    findings = run_all(ds, only=["polygon_mask_consistency"])
    assert len(findings) == 1
    iou = findings[0].evidence["rasterized_iou"]
    assert iou < 0.9


def test_polygon_mask_consistency_passes_when_aligned() -> None:
    poly = Polygon(points=((10.0, 10.0), (80.0, 10.0), (80.0, 80.0), (10.0, 80.0)))
    mask_arr = np.zeros((100, 100), dtype=bool)
    mask_arr[10:81, 10:81] = True  # matches polygon tightly

    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            _ann(0, "a.jpg", 0, "leaf", AnnotationKind.POLYGON, poly),
            _ann(1, "a.jpg", 0, "leaf", AnnotationKind.MASK, Mask(array=mask_arr)),
        ],
        categories=[Category(id=0, name="leaf")],
    )
    findings = run_all(ds, only=["polygon_mask_consistency"])
    assert findings == []


# ---- Split purity ----

def test_split_purity_fires_on_skewed_splits() -> None:
    # train: 100 cars, 100 trucks. val: 100 cars, 0 trucks. -> p ~ 0.
    anns = []
    for i in range(100):
        anns.append(_ann(len(anns), f"t_car_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10)))
        anns.append(_ann(len(anns), f"t_trk_{i}.jpg", 1, "trk", AnnotationKind.BOX, Box(0, 0, 10, 10)))
    for i in range(100):
        anns.append(_ann(len(anns), f"v_car_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10)))
    images = [Image(filename=n, width=100, height=100) for n in {a.image_filename for a in anns}]
    splits = {}
    for a in anns:
        splits[a.image_filename] = "train" if a.image_filename.startswith("t_") else "val"
    ds = Dataset(
        images=images,
        annotations=anns,
        categories=[Category(id=0, name="car"), Category(id=1, name="trk")],
        split_by_filename=splits,
    )
    findings = run_all(ds, only=["split_purity"])
    assert len(findings) == 1
    assert findings[0].evidence["p_value"] < 0.05


def test_split_purity_passes_on_balanced_splits() -> None:
    anns = []
    for i in range(50):
        anns.append(_ann(len(anns), f"t_car_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10)))
        anns.append(_ann(len(anns), f"t_trk_{i}.jpg", 1, "trk", AnnotationKind.BOX, Box(0, 0, 10, 10)))
        anns.append(_ann(len(anns), f"v_car_{i}.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10)))
        anns.append(_ann(len(anns), f"v_trk_{i}.jpg", 1, "trk", AnnotationKind.BOX, Box(0, 0, 10, 10)))
    images = [Image(filename=n, width=100, height=100) for n in {a.image_filename for a in anns}]
    splits = {
        a.image_filename: "train" if a.image_filename.startswith("t_") else "val"
        for a in anns
    }
    ds = Dataset(
        images=images,
        annotations=anns,
        categories=[Category(id=0, name="car"), Category(id=1, name="trk")],
        split_by_filename=splits,
    )
    findings = run_all(ds, only=["split_purity"])
    assert findings == []


def test_split_purity_skipped_without_splits() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[_ann(0, "a.jpg", 0, "car", AnnotationKind.BOX, Box(0, 0, 10, 10))],
        categories=[Category(id=0, name="car")],
    )
    findings = run_all(ds, only=["split_purity"])
    assert findings == []


# ---- Split leakage ----

def test_split_leakage_skipped_without_imagehash(tmp_path) -> None:
    import importlib.util
    if importlib.util.find_spec("imagehash") is not None:
        pytest.skip("imagehash installed; no-op path not reachable")
    ds = Dataset(
        images=[Image(filename="a.jpg", width=10, height=10, path=tmp_path / "a.jpg")],
        annotations=[],
        categories=[],
        split_by_filename={"a.jpg": "train"},
    )
    assert run_all(ds, only=["split_leakage"]) == []
