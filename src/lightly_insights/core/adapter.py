"""Adapter from the existing analyze.* result objects to a Dataset.

The existing pipeline stores per-class aggregates (sizes, heatmaps, etc.)
but does NOT keep the raw per-annotation list by design (memory). The
adapter therefore re-reads the labels via the supplied ``label_input`` —
one extra pass over the (already-in-memory) objects — and produces a
fully-populated ``Dataset`` ready for any ``Check``.

This lets the new check-based pipeline run side-by-side with the legacy
analyze/present flow during the migration.
"""
from __future__ import annotations

from typing import Optional

from labelformat.model.object_detection import ObjectDetectionInput

from lightly_insights.analyze import (
    ImageAnalysis,
    ObjectDetectionAnalysis,
)
from lightly_insights.core.dataset import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Image,
)
from lightly_insights.core.geometry import Box


def build_dataset(
    image_analysis: ImageAnalysis,
    od_analysis: Optional[ObjectDetectionAnalysis] = None,
    label_input: Optional[ObjectDetectionInput] = None,
) -> Dataset:
    """Assemble a Dataset from the existing analysis objects.

    ``label_input`` is required if you want per-annotation checks to work.
    Without it we still build a valid Dataset but the ``annotations`` list
    is empty — image-level checks still run.
    """
    images = [
        Image(
            filename=name,
            width=w,
            height=h,
            path=image_analysis.image_folder / name,
        )
        for name, (w, h) in image_analysis.filename_to_size.items()
    ]

    categories: list = []
    if od_analysis is not None:
        categories = [
            Category(id=cid, name=c.class_name)
            for cid, c in od_analysis.classes.items()
        ]

    annotations: list = []
    if label_input is not None:
        ann_id = 0
        for label in label_input.get_labels():
            for obj in label.objects:
                annotations.append(
                    Annotation(
                        annotation_id=ann_id,
                        image_filename=label.image.filename,
                        class_id=obj.category.id,
                        class_name=obj.category.name,
                        kind=AnnotationKind.BOX,
                        geometry=Box(
                            xmin=obj.box.xmin,
                            ymin=obj.box.ymin,
                            xmax=obj.box.xmax,
                            ymax=obj.box.ymax,
                        ),
                    )
                )
                ann_id += 1

    return Dataset(
        images=images,
        annotations=annotations,
        categories=categories,
        corrupt_filenames=list(image_analysis.corrupt_files),
    )
