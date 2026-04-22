"""Too-few-backgrounds check.

If a detection dataset has no (or almost no) images with zero annotations,
the model never learns "nothing is here" and over-predicts at inference.
Detection folklore recommends 5-10 % pure backgrounds minimum.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity

MIN_BACKGROUND_FRACTION = 0.05  # 5 % of images should have zero annotations


@register_check
class BackgroundScarcityCheck(Check):
    check_id = "background_scarcity"
    title = "Too few background images"
    category = "balance"
    # No annotation-kind restriction — we just count images without annotations.

    def run(self, dataset: Dataset) -> List[Finding]:
        if dataset.num_images == 0:
            return []
        # Don't emit for label-free datasets; the finding would be meaningless.
        if not dataset.annotations:
            return []
        annotated = set(dataset.annotations_by_image.keys())
        zero_ann = [img.filename for img in dataset.images if img.filename not in annotated]
        bg_fraction = len(zero_ann) / dataset.num_images
        if bg_fraction >= MIN_BACKGROUND_FRACTION:
            return []
        return [
            Finding(
                check_id=self.check_id,
                severity=Severity.MEDIUM,
                category=self.category,
                title="Background images are scarce",
                detail=(
                    f"Only {len(zero_ann)} of {dataset.num_images} images "
                    f"({100 * bg_fraction:.1f} %) contain zero annotations. "
                    "Detectors trained without pure backgrounds tend to "
                    "over-predict at inference."
                ),
                action=(
                    f"Add unlabeled-but-valid images until the background "
                    f"fraction is \u2265 {int(100 * MIN_BACKGROUND_FRACTION)} %."
                ),
                # A few examples of already-zero-annotation images (if any)
                # so the reporter can show them.
                affected_images=sorted(zero_ann)[:5],
                evidence={
                    "background_fraction": round(bg_fraction, 4),
                    "background_count": len(zero_ann),
                    "total_images": dataset.num_images,
                    "threshold": MIN_BACKGROUND_FRACTION,
                },
            )
        ]
