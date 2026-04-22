"""Polygon ↔ mask drift detection.

When a dataset carries both polygon and mask representations of the same
instance (common in panoptic / labeler-export workflows), they should
agree. They often don't — a labeler edits the polygon but the rasterized
mask doesn't get re-exported, or vice versa.

Pairing heuristic: for each image, polygons and masks of the same class
whose tight bounding boxes overlap with IoU ≥ PAIR_IOU are considered to
refer to the same instance. For each pair, rasterize the polygon at the
image resolution and compute mask IoU. Disagreements below
``AGREEMENT_IOU`` are flagged.

The check silently no-ops on datasets that don't carry both kinds.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Annotation, AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.geometry import Mask, Polygon

# Tight-box IoU for considering a polygon and a mask to be the same instance.
PAIR_IOU = 0.9
# Polygon-rasterized IoU with the supplied mask. Below this = drift.
AGREEMENT_IOU = 0.9


@register_check
class PolygonMaskConsistencyCheck(Check):
    check_id = "polygon_mask_consistency"
    title = "Polygon and mask disagree"
    category = "annotation"
    supported_kinds = frozenset({AnnotationKind.POLYGON, AnnotationKind.MASK})

    def applies_to(self, dataset: Dataset) -> bool:
        # Needs BOTH polygon and mask annotations to have anything to compare.
        return (
            AnnotationKind.POLYGON in dataset.kinds
            and AnnotationKind.MASK in dataset.kinds
        )

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        image_by_name = dataset.image_by_filename

        for filename, anns in dataset.annotations_by_image.items():
            polys = [a for a in anns if a.kind == AnnotationKind.POLYGON]
            masks = [a for a in anns if a.kind == AnnotationKind.MASK]
            if not polys or not masks:
                continue

            image = image_by_name.get(filename)
            if image is None or image.width <= 0 or image.height <= 0:
                continue

            # Greedy pair polygons to masks: for each polygon, pick the
            # highest-IoU same-class mask, pair once.
            used_masks: set = set()
            for poly_ann in polys:
                best_iou = 0.0
                best_mask: Optional[Annotation] = None
                for mask_ann in masks:
                    if mask_ann.annotation_id in used_masks:
                        continue
                    if mask_ann.class_id != poly_ann.class_id:
                        continue
                    iou = poly_ann.tight_box.iou(mask_ann.tight_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = mask_ann
                if best_mask is None or best_iou < PAIR_IOU:
                    continue
                used_masks.add(best_mask.annotation_id)

                # Rasterize the polygon and compare.
                poly = poly_ann.geometry
                mask = best_mask.geometry
                if not isinstance(poly, Polygon) or not isinstance(mask, Mask):
                    continue
                rasterized = _rasterize_polygon(
                    poly, width=image.width, height=image.height
                )
                # Resize mask if needed — we only trust exact-size masks but
                # the mask_image_size_mismatch check owns that warning.
                if rasterized.shape != mask.array.shape:
                    continue
                iou = _mask_iou(rasterized, mask.array)
                if iou >= AGREEMENT_IOU:
                    continue
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        severity=Severity.HIGH,
                        category=self.category,
                        title=(
                            f"Polygon/mask drift for class '{poly_ann.class_name}'"
                        ),
                        detail=(
                            f"Polygon and mask on {filename} refer to the "
                            f"same '{poly_ann.class_name}' instance "
                            f"(tight-box IoU {best_iou:.2f}) but their "
                            f"rasterized IoU is {iou:.2f}. One of the two "
                            "is stale."
                        ),
                        action=(
                            "Re-export the mask from the polygon (or vice "
                            "versa) so both representations match."
                        ),
                        affected_images=[filename],
                        affected_annotations=[
                            poly_ann.annotation_id,
                            best_mask.annotation_id,
                        ],
                        evidence={
                            "rasterized_iou": round(iou, 3),
                            "tight_box_iou": round(best_iou, 3),
                            "class_name": poly_ann.class_name,
                        },
                    )
                )
        return findings


def _rasterize_polygon(
    poly: Polygon, width: int, height: int
) -> "np.ndarray":
    """Rasterize a polygon to a bool mask at the image resolution.

    Uses PIL (already a dependency); avoids pulling in shapely/skimage.
    """
    from PIL import Image as PILImage
    from PIL import ImageDraw

    img = PILImage.new("L", (width, height), 0)
    if len(poly.points) >= 3:
        draw = ImageDraw.Draw(img)
        # PIL expects [(x, y), ...] with integer coords accepted as float.
        draw.polygon([tuple(p) for p in poly.points], outline=1, fill=1)
    return np.asarray(img, dtype=bool)


def _mask_iou(a: "np.ndarray", b: "np.ndarray") -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0  # both empty; trivially agree
    return inter / union
