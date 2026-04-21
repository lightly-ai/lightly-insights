"""Regression tests for Phase 1 correctness bugs.

Each test targets a specific P0 item from the audit report. Tests are short
and use synthetic fixtures so they can run without real datasets.
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
