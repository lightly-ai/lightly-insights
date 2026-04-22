"""Regression tests for Phase 1 correctness bugs and Phase 2 new insights.

Each test targets a specific audit item. Tests are short and use synthetic
fixtures so they can run without real datasets.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Iterable, List

import numpy as np
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)
from PIL import Image as PILImage

from lightly_insights import analyze, plots, present


@dataclass
class _FakeODInput(ObjectDetectionInput):
    categories: List[Category]
    labels: List[ImageObjectDetection]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:  # pragma: no cover
        pass

    def get_categories(self) -> Iterable[Category]:
        return list(self.categories)

    def get_images(self) -> Iterable[Image]:
        return [label.image for label in self.labels]

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        return list(self.labels)


def _make_image_folder(tmp_path: Path, sizes: List[tuple]) -> Path:
    folder = tmp_path / "imgs"
    folder.mkdir()
    for idx, (w, h) in enumerate(sizes):
        PILImage.new("RGB", (w, h), color=(0, 0, 0)).save(folder / f"img_{idx}.png")
    return folder


# ---- 1.1 median_size ----

def test_median_size_is_true_median(tmp_path: Path) -> None:
    # Even count: median of widths [100, 200, 300, 400] should be 250.
    folder = _make_image_folder(
        tmp_path, [(100, 400), (200, 300), (300, 200), (400, 100)]
    )
    result = analyze.analyze_images(folder)
    assert result.median_size == (250, 250)


def test_median_size_odd_count(tmp_path: Path) -> None:
    folder = _make_image_folder(tmp_path, [(100, 400), (300, 200), (500, 600)])
    result = analyze.analyze_images(folder)
    assert result.median_size == (300, 400)


def test_median_size_empty_folder(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    result = analyze.analyze_images(folder)
    assert result.median_size == (0, 0)


# ---- 1.2 objects_per_image counts images, not objects ----

def test_objects_per_image_counts_images_not_objects() -> None:
    cat = Category(id=0, name="thing")
    img_a = Image(id=0, filename="a.jpg", width=100, height=100)
    img_b = Image(id=1, filename="b.jpg", width=100, height=100)
    img_c = Image(id=2, filename="c.jpg", width=100, height=100)

    def _obj() -> SingleObjectDetection:
        return SingleObjectDetection(
            category=cat, box=BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
        )

    label_input = _FakeODInput(
        categories=[cat],
        labels=[
            ImageObjectDetection(image=img_a, objects=[_obj(), _obj(), _obj()]),
            ImageObjectDetection(image=img_b, objects=[_obj(), _obj(), _obj()]),
            ImageObjectDetection(image=img_c, objects=[_obj()]),
        ],
    )
    analysis = analyze.analyze_object_detections(label_input)
    # Two images contain 3 objects, one image contains 1 object.
    assert analysis.classes[0].objects_per_image == Counter({3: 2, 1: 1})


# ---- 1.3 side_length_avg formula + xlabel ----

def test_side_length_average_formula(tmp_path: Path, mocker) -> None:  # type: ignore[no-untyped-def]
    captured = {}

    def _fake_histogram(**kwargs: object) -> None:
        # Capture the call that uses the side-length title.
        title = kwargs.get("title", "")
        assert isinstance(title, str)
        if "Side Length" in title:
            captured["hist"] = kwargs["hist"]
            captured["xlabel"] = kwargs["xlabel"]

    mocker.patch.object(plots, "_histogram", side_effect=_fake_histogram)
    mocker.patch.object(plots, "_heatmap")
    mocker.patch.object(plots, "width_heigth_pixels_plot")
    mocker.patch.object(plots, "_width_heigth_percent_plot")

    class_analysis = analyze.ClassAnalysis.create_empty(id=0, name="c")
    # Single box: width=100, height=200. (w+h)/2 = 150 -> bucket 150.
    # With the old (buggy) formula w + h/2 = 200 -> bucket 200.
    class_analysis.object_sizes_abs.append((100.0, 200.0))
    class_analysis.object_sizes_rel.append((0.5, 0.5))

    plot_folder = tmp_path / "plots"
    plot_folder.mkdir()
    plots.create_object_plots(
        output_folder=tmp_path, plot_folder=plot_folder, class_analysis=class_analysis
    )

    assert captured["hist"] == Counter({150.0: 1})
    assert captured["xlabel"] == "(Width + Height) / 2 (px)"


# ---- 1.8 heatmap: sub-cell boxes still contribute ----

def test_heatmap_tiny_boxes_contribute() -> None:
    cat = Category(id=0, name="thing")
    img = Image(id=0, filename="a.jpg", width=1000, height=1000)
    # Box spans pixels 32..38 x 32..38. With HEATMAP_SIZE=100 that's
    # grid cells 3.2..3.8 — entirely inside cell (3, 3). The buggy
    # implementation produced an empty slice.
    tiny = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=32, ymin=32, xmax=38, ymax=38)
    )
    label_input = _FakeODInput(
        categories=[cat],
        labels=[ImageObjectDetection(image=img, objects=[tiny])],
    )
    analysis = analyze.analyze_object_detections(label_input)
    assert analysis.total.heatmap.sum() > 0
    assert analysis.classes[0].heatmap.sum() > 0


# ---- 1.9 filename matching ignores extensions / subdirs ----

def test_filename_matching_ignores_extensions(tmp_path: Path) -> None:
    # Images are listed as "foo.jpg"; labels arrive as "sub/foo.xml".
    image_set = {"foo.jpg", "bar.jpg", "only_image.jpg"}
    label_set = {"sub/foo.xml", "bar.xml", "only_label.xml"}
    result = present._get_filename_insights(
        output_folder=tmp_path,
        image_filename_set=image_set,
        label_filename_set=label_set,
    )
    assert result.num_images_no_label == 1
    assert result.sample_filenames_no_label == ["only_image.jpg"]
    assert result.num_labels_no_image == 1
    assert result.sample_filenames_no_image == ["only_label.xml"]


# ==== Phase 2 ====


# ---- 2.3 corrupt-image handling ----

def test_analyze_images_skips_corrupt_files(tmp_path: Path) -> None:
    folder = _make_image_folder(tmp_path, [(100, 100), (200, 200)])
    # Create a file with a valid image extension but garbage contents.
    (folder / "broken.png").write_bytes(b"not-an-image")
    result = analyze.analyze_images(folder)
    assert result.num_images == 2
    assert "broken.png" in result.corrupt_files


# ---- 3.3 tiny / huge object flags ----

def test_tiny_and_huge_object_counts() -> None:
    cat = Category(id=0, name="thing")
    img = Image(id=0, filename="a.jpg", width=1000, height=1000)
    tiny = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=10, ymin=10, xmax=30, ymax=30)
    )  # 20*20 / 1e6 = 0.04% -> tiny
    huge = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=100, ymin=100, xmax=900, ymax=900)
    )  # 800*800 / 1e6 = 64% -> huge
    normal = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=400, ymin=400, xmax=500, ymax=500)
    )  # 10000 / 1e6 = 1% -> neither
    label_input = _FakeODInput(
        categories=[cat],
        labels=[ImageObjectDetection(image=img, objects=[tiny, huge, normal])],
    )
    result = analyze.analyze_object_detections(label_input)
    assert result.classes[0].tiny_object_count == 1
    assert result.classes[0].huge_object_count == 1
    assert result.total.tiny_object_count == 1
    assert result.total.huge_object_count == 1


# ---- 3.4 edge-touching ----

def test_edge_touching_detected() -> None:
    cat = Category(id=0, name="thing")
    img = Image(id=0, filename="a.jpg", width=1000, height=1000)
    left_edge = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=0, ymin=100, xmax=50, ymax=200)
    )
    right_edge = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=500, ymin=500, xmax=1000, ymax=600)
    )
    interior = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=200, ymin=200, xmax=400, ymax=400)
    )
    label_input = _FakeODInput(
        categories=[cat],
        labels=[
            ImageObjectDetection(
                image=img, objects=[left_edge, right_edge, interior]
            )
        ],
    )
    result = analyze.analyze_object_detections(label_input)
    assert result.classes[0].edge_touching_count == 2


# ---- 3.6 duplicate-annotation detection ----

def test_duplicate_annotation_detection() -> None:
    cat = Category(id=0, name="thing")
    img = Image(id=0, filename="a.jpg", width=1000, height=1000)
    a = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=100, ymin=100, xmax=200, ymax=200)
    )
    # Near-identical (IoU ~ 0.92) -> flagged
    a_dup = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=102, ymin=102, xmax=202, ymax=202)
    )
    # Completely separate -> not flagged
    b = SingleObjectDetection(
        category=cat, box=BoundingBox(xmin=500, ymin=500, xmax=600, ymax=600)
    )
    label_input = _FakeODInput(
        categories=[cat],
        labels=[ImageObjectDetection(image=img, objects=[a, a_dup, b])],
    )
    result = analyze.analyze_object_detections(label_input)
    assert len(result.duplicate_annotations) == 1
    pair = result.duplicate_annotations[0]
    assert pair.filename == "a.jpg"
    assert pair.iou >= 0.9


def test_box_iou_helpers() -> None:
    box_a = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)
    # Identical.
    assert analyze._box_iou(box_a, box_a) == 1.0
    # No overlap.
    box_far = BoundingBox(xmin=100, ymin=100, xmax=110, ymax=110)
    assert analyze._box_iou(box_a, box_far) == 0.0
    # Half-overlap: (5x10) / (10x10 + 5x10 - 5x10) = 50/100 = 0.5.
    box_half = BoundingBox(xmin=5, ymin=0, xmax=15, ymax=10)
    assert abs(analyze._box_iou(box_a, box_half) - (50 / 150)) < 1e-9


# ---- 3.1 class imbalance ----

def test_imbalance_balanced() -> None:
    stats = present._compute_imbalance_stats([100, 100, 100, 100])
    assert abs(stats.normalized_entropy - 1.0) < 1e-9
    assert abs(stats.gini) < 1e-9
    assert abs(stats.top_class_share - 0.25) < 1e-9
    assert stats.under_represented_count == 0


def test_imbalance_skewed() -> None:
    # 95% in one class, four with 1% each.
    stats = present._compute_imbalance_stats([95, 2, 1, 1, 1])
    assert stats.normalized_entropy < 0.5
    assert stats.gini > 0.5
    assert abs(stats.top_class_share - 0.95) < 1e-9
    # 1% exactly is not strictly less than 1% -> not counted.
    # Only the 2% slice is not under, and the three 1% ones tie.
    # We use `<` (strict), so 1/100 = 0.01 fails the `< 0.01` check -> 0 flagged.
    # But 0 and empty-class counts would count. Here all classes > 0.


def test_imbalance_empty() -> None:
    stats = present._compute_imbalance_stats([])
    assert stats.normalized_entropy == 0.0
    assert stats.gini == 0.0
    assert stats.top_class_share == 0.0


def test_imbalance_single_class() -> None:
    stats = present._compute_imbalance_stats([42])
    # With a single non-zero class entropy is 0, gini is 0, top_share 1.0.
    assert stats.normalized_entropy == 0.0
    assert abs(stats.top_class_share - 1.0) < 1e-9
    assert stats.gini == 0.0


# ==== Phase 3 ====


# ---- 3.5 co-occurrence matrix ----

def test_cooccurrence_matrix_counts_joint_images() -> None:
    cats = [
        Category(id=0, name="a"),
        Category(id=1, name="b"),
        Category(id=2, name="c"),
    ]
    img = lambda i: Image(id=i, filename=f"img_{i}.jpg", width=100, height=100)  # noqa: E731
    box = BoundingBox(xmin=0, ymin=0, xmax=10, ymax=10)

    def obj(cat: Category) -> SingleObjectDetection:
        return SingleObjectDetection(category=cat, box=box)

    label_input = _FakeODInput(
        categories=cats,
        labels=[
            ImageObjectDetection(image=img(0), objects=[obj(cats[0]), obj(cats[1])]),
            ImageObjectDetection(image=img(1), objects=[obj(cats[0]), obj(cats[1])]),
            ImageObjectDetection(image=img(2), objects=[obj(cats[1]), obj(cats[2])]),
            ImageObjectDetection(image=img(3), objects=[obj(cats[0])]),
        ],
    )
    result = analyze.analyze_object_detections(label_input)
    m = result.cooccurrence_matrix
    assert m is not None
    # Diagonals: images containing each class.
    # a appears in img 0,1,3 -> 3
    # b appears in img 0,1,2 -> 3
    # c appears in img 2      -> 1
    assert m[0, 0] == 3
    assert m[1, 1] == 3
    assert m[2, 2] == 1
    # Off-diagonal: a+b together in img 0,1 -> 2; b+c in img 2 -> 1; a+c none.
    assert m[0, 1] == 2 and m[1, 0] == 2
    assert m[1, 2] == 1 and m[2, 1] == 1
    assert m[0, 2] == 0 and m[2, 0] == 0


# ---- 2.8 curated sample grid ----

def test_select_sample_images_handles_small_datasets(tmp_path: Path) -> None:
    # 3 images: fewer than target_count=8, must not crash.
    folder = _make_image_folder(tmp_path, [(100, 100), (200, 200), (300, 300)])
    image_analysis = analyze.analyze_images(folder)
    # Build a trivial empty OD analysis.
    empty_od = analyze.ObjectDetectionAnalysis(
        num_images=0,
        num_images_zero_objects=0,
        filename_set=set(),
        total=analyze.ClassAnalysis.create_empty(id=-1, name="[All classes]"),
        classes={},
    )
    picks = present._select_sample_images(
        image_analysis=image_analysis, od_analysis=empty_od
    )
    # Should return at most 3 samples, all unique, no crash.
    assert len(picks) <= 3
    assert len(set(p.filename for p in picks)) == len(picks)


def test_select_sample_images_curates_edge_cases(tmp_path: Path) -> None:
    folder = _make_image_folder(tmp_path, [(50, 50), (400, 400), (2000, 2000)])
    image_analysis = analyze.analyze_images(folder)
    empty_od = analyze.ObjectDetectionAnalysis(
        num_images=0,
        num_images_zero_objects=0,
        filename_set=set(),
        total=analyze.ClassAnalysis.create_empty(id=-1, name="[All classes]"),
        classes={},
    )
    picks = present._select_sample_images(
        image_analysis=image_analysis, od_analysis=empty_od
    )
    labels = {p.label for p in picks}
    # Smallest + largest should both be represented.
    assert "smallest" in labels
    assert "largest" in labels


# ---- 2.2 recursive listing ----

def test_recursive_listing_traverses_subdirs(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    PILImage.new("RGB", (100, 100)).save(tmp_path / "train" / "a.jpg")
    PILImage.new("RGB", (100, 100)).save(tmp_path / "val" / "b.jpg")
    # A top-level scan finds nothing (no images at the root).
    flat = analyze.analyze_images(tmp_path, recursive=False)
    assert flat.num_images == 0
    # Recursive pass finds both.
    deep = analyze.analyze_images(tmp_path, recursive=True)
    assert deep.num_images == 2
    # Filenames keep the subdir prefix so sub/a.jpg and sub2/a.jpg wouldn't collide.
    assert "train/a.jpg" in deep.filename_set
    assert "val/b.jpg" in deep.filename_set


# ---- 3.8 quality red flags ----

def test_quality_flags_detect_uniform_image(tmp_path: Path) -> None:
    folder = tmp_path / "imgs"
    folder.mkdir()
    # All-black uniform image.
    PILImage.new("RGB", (200, 200), color=(0, 0, 0)).save(folder / "black.png")
    # Normal content image with varying pixels.
    import numpy as np
    arr = (np.random.RandomState(0).rand(200, 200, 3) * 255).astype("uint8")
    PILImage.fromarray(arr).save(folder / "noisy.png")
    # Extreme aspect ratio.
    PILImage.new("RGB", (2000, 100), color=(128, 128, 128)).save(folder / "wide.png")

    result = analyze.analyze_images(folder, check_quality=True)
    flags = result.quality_flags
    assert flags is not None
    assert "black.png" in flags.uniform_files
    assert "wide.png" in flags.extreme_aspect_files


def test_quality_flags_skipped_when_disabled(tmp_path: Path) -> None:
    folder = _make_image_folder(tmp_path, [(100, 100)])
    result = analyze.analyze_images(folder, check_quality=False)
    assert result.quality_flags is None


# ---- 3.7 near-duplicate detection gracefully no-ops without imagehash ----

def test_near_duplicates_empty_when_disabled(tmp_path: Path) -> None:
    folder = _make_image_folder(tmp_path, [(100, 100), (200, 200)])
    result = analyze.analyze_images(folder, find_near_duplicates=False)
    assert result.near_duplicate_groups == []


# ---- 4.3 CLI wrapper ----

def test_cli_images_only(tmp_path: Path) -> None:
    from lightly_insights import __main__ as cli

    # Need at least 1 image so analyze_images has work.
    img_folder = _make_image_folder(tmp_path, [(100, 100), (200, 200)])
    out = tmp_path / "report"
    rc = cli.main([
        "--images", str(img_folder),
        "--out", str(out),
        "--images-only",
        "--no-quality-check",
    ])
    assert rc == 0
    assert (out / "index.html").exists()
