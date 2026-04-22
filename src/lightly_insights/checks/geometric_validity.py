"""Geometric validity checks.

Three "did this even parse" checks that should fire before anything else
looks at the data:

- ``DegenerateAnnotationCheck`` — annotations with zero/negative area,
  malformed polygons, empty masks. Usually an export bug.
- ``OutOfBoundsAnnotationCheck`` — annotations that extend beyond the
  image they belong to. Usually a resize/crop that forgot to clip.
- ``MaskImageSizeMismatchCheck`` — masks whose array shape doesn't match
  the image dimensions. Silent disaster at training time.

Every downstream check assumes the geometry is valid, so these run early
and emit findings for every offender.
"""
from __future__ import annotations

from typing import List, Tuple

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Box, Mask, Polygon


@register_check
class DegenerateAnnotationCheck(Check):
    check_id = "degenerate_annotation"
    title = "Degenerate annotation"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for ann in dataset.annotations:
            reason = _degenerate_reason(ann.geometry)
            if reason is None:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.CRITICAL,
                    category=self.category,
                    title=f"Degenerate annotation for class '{ann.class_name}'",
                    detail=(
                        f"Annotation on {ann.image_filename} is degenerate: "
                        f"{reason}. Export or upstream processing is likely broken."
                    ),
                    action="Remove or fix the annotation before training.",
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={"reason": reason, "kind": ann.kind.value},
                )
            )
        return findings


def _degenerate_reason(geometry: object) -> "str | None":
    if isinstance(geometry, Box):
        if geometry.xmax <= geometry.xmin:
            return "xmax <= xmin"
        if geometry.ymax <= geometry.ymin:
            return "ymax <= ymin"
        return None
    if isinstance(geometry, Polygon):
        if len(geometry.points) < 3:
            return f"polygon has {len(geometry.points)} points (< 3)"
        if geometry.area == 0:
            return "polygon has zero area (collinear points)"
        return None
    if isinstance(geometry, Mask):
        if geometry.array.size == 0:
            return "mask array is empty"
        if geometry.area == 0:
            return "mask has no foreground pixels"
        return None
    return None


@register_check
class OutOfBoundsAnnotationCheck(Check):
    check_id = "out_of_bounds_annotation"
    title = "Annotation extends outside image"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        image_by_name = dataset.image_by_filename
        for ann in dataset.annotations:
            image = image_by_name.get(ann.image_filename)
            if image is None or image.width <= 0 or image.height <= 0:
                # Nothing to compare against; other checks handle corrupt files.
                continue
            overrun = _out_of_bounds_overrun(ann.geometry, image.width, image.height)
            if overrun is None:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.HIGH,
                    category=self.category,
                    title=(
                        f"Out-of-bounds annotation for class '{ann.class_name}'"
                    ),
                    detail=(
                        f"Annotation on {ann.image_filename} extends "
                        f"{overrun:.1f} px beyond the image boundary "
                        f"({image.width}\u00d7{image.height}). Typically caused "
                        "by a resize or crop that didn't clip labels."
                    ),
                    action="Clip the annotation to image bounds or drop it.",
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "overrun_pixels": round(overrun, 2),
                        "image_width": image.width,
                        "image_height": image.height,
                    },
                )
            )
        return findings


def _out_of_bounds_overrun(geometry: object, w: int, h: int) -> "float | None":
    """How many pixels does the geometry extend past the image rect?

    Returns ``None`` when the geometry is fully inside (with a 0.5 px tolerance
    to avoid floating-point false positives). Otherwise returns the worst-case
    overshoot across all four edges.
    """
    tol = 0.5
    if isinstance(geometry, Box):
        bb = geometry
    elif isinstance(geometry, Polygon):
        if not geometry.points:
            return None
        bb = geometry.tight_box
    elif isinstance(geometry, Mask):
        if geometry.array.shape[1] > w or geometry.array.shape[0] > h:
            return float(
                max(geometry.array.shape[1] - w, geometry.array.shape[0] - h)
            )
        return None
    else:
        return None
    overruns = [
        -bb.xmin,
        -bb.ymin,
        bb.xmax - w,
        bb.ymax - h,
    ]
    worst = max(overruns)
    return worst if worst > tol else None


@register_check
class MaskImageSizeMismatchCheck(Check):
    check_id = "mask_image_size_mismatch"
    title = "Mask shape doesn't match image"
    category = "annotation"
    supported_kinds = frozenset({AnnotationKind.MASK})

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        image_by_name = dataset.image_by_filename
        for ann in dataset.annotations:
            if ann.kind != AnnotationKind.MASK:
                continue
            mask = ann.geometry
            if not isinstance(mask, Mask):
                continue
            image = image_by_name.get(ann.image_filename)
            if image is None or image.width <= 0 or image.height <= 0:
                continue
            mh, mw = mask.array.shape[:2]
            if mh == image.height and mw == image.width:
                continue
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.CRITICAL,
                    category=self.category,
                    title=(
                        f"Mask size mismatch for class '{ann.class_name}'"
                    ),
                    detail=(
                        f"Mask on {ann.image_filename} is "
                        f"{mw}\u00d7{mh} but image is "
                        f"{image.width}\u00d7{image.height}. Training will "
                        "learn on misaligned data until this is fixed."
                    ),
                    action=(
                        "Re-rasterize the mask at the image resolution, or "
                        "resize the image to match the mask."
                    ),
                    affected_images=[ann.image_filename],
                    affected_annotations=[ann.annotation_id],
                    evidence={
                        "mask_width": mw,
                        "mask_height": mh,
                        "image_width": image.width,
                        "image_height": image.height,
                    },
                )
            )
        return findings
