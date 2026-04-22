"""Flag annotations whose shape is wildly different from the class median.

For each class we compute the median and MAD (median absolute deviation) of
area and aspect ratio, then flag annotations more than ``MAD_THRESHOLD``
MADs from either median. MAD is robust to the outliers we're looking for —
a 3σ rule on raw std would drag with the outlier and underflag.

Works for every annotation kind because we use ``geometry.area`` and the
``tight_box`` aspect ratio, both of which every geometry exposes.

This is one check that catches:
- the 20-pixel "car" in a dataset of 400-pixel cars
- the full-image "person" bounding box
- the stretched 10:1 box that should have been 2:1
- class-label copy-paste errors between very different shapes
"""
from __future__ import annotations

from statistics import median
from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

MAD_THRESHOLD = 5.0  # flag outliers beyond this many MADs from class median
MIN_CLASS_SAMPLES = 20  # below this, medians aren't meaningful


def _mad(values: List[float], med: float) -> float:
    """Median absolute deviation. 0.0 if identical values (caller checks)."""
    if not values:
        return 0.0
    return median(abs(v - med) for v in values)


@register_check
class ShapeOutlierCheck(Check):
    check_id = "shape_outlier"
    title = "Annotation shape outlier"
    category = "annotation"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        for cat in dataset.categories:
            class_anns = dataset.annotations_by_class.get(cat.id, [])
            if len(class_anns) < MIN_CLASS_SAMPLES:
                # Not enough data to flag outliers meaningfully.
                continue

            areas = [ann.area for ann in class_anns]
            ratios = [
                ann.tight_box.aspect_ratio
                for ann in class_anns
                if ann.tight_box.height > 0
            ]
            if not areas or not ratios:
                continue

            med_area = median(areas)
            mad_area = _mad(areas, med_area)
            med_ratio = median(ratios)
            mad_ratio = _mad(ratios, med_ratio)

            for ann in class_anns:
                reasons = []

                # Area outlier. When MAD is zero the class is perfectly
                # uniform; any deviation is an outlier by definition.
                if mad_area > 0:
                    area_dev = abs(ann.area - med_area) / mad_area
                    if area_dev > MAD_THRESHOLD:
                        direction = "large" if ann.area > med_area else "small"
                        reasons.append(
                            f"area {ann.area:.0f} is {area_dev:.1f}\u00d7 MAD "
                            f"from class median {med_area:.0f} (too {direction})"
                        )
                elif ann.area != med_area:
                    direction = "large" if ann.area > med_area else "small"
                    reasons.append(
                        f"area {ann.area:.0f} differs from a perfectly "
                        f"uniform class median {med_area:.0f} (too {direction})"
                    )

                # Aspect ratio outlier.
                if ann.tight_box.height > 0:
                    ratio = ann.tight_box.aspect_ratio
                    if mad_ratio > 0:
                        ratio_dev = abs(ratio - med_ratio) / mad_ratio
                        if ratio_dev > MAD_THRESHOLD:
                            reasons.append(
                                f"aspect {ratio:.2f} is {ratio_dev:.1f}\u00d7 "
                                f"MAD from class median {med_ratio:.2f}"
                            )
                    elif abs(ratio - med_ratio) > 0.01:
                        reasons.append(
                            f"aspect {ratio:.2f} differs from a perfectly "
                            f"uniform class median {med_ratio:.2f}"
                        )

                if not reasons:
                    continue

                findings.append(
                    Finding(
                        check_id=self.check_id,
                        severity=Severity.MEDIUM,
                        category=self.category,
                        title=(
                            f"Outlier annotation for class '{ann.class_name}'"
                        ),
                        detail=(
                            f"Annotation on {ann.image_filename} is an outlier: "
                            + "; ".join(reasons) + "."
                        ),
                        action=(
                            "Compare against other samples of this class; "
                            "check for a class-label copy-paste error or a "
                            "wildly miscropped box."
                        ),
                        affected_images=[ann.image_filename],
                        affected_annotations=[ann.annotation_id],
                        evidence={
                            "class_id": cat.id,
                            "class_name": cat.name,
                            "area": round(ann.area, 2),
                            "median_area": round(med_area, 2),
                            "aspect_ratio": round(ann.tight_box.aspect_ratio, 3),
                            "median_aspect": round(med_ratio, 3),
                        },
                    )
                )
        return findings
