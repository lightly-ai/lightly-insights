"""Opinionated training-readiness advice derived from dataset stats.

Each rule has three parts:
  - a trigger condition over existing analysis fields
  - a one-line human-readable finding
  - a concrete suggestion the user can act on

These are *guidelines*, not guarantees. They're tuned for YOLO-family
detectors since that's the most common workload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover
    from lightly_insights.analyze import ImageAnalysis, ObjectDetectionAnalysis


@dataclass(frozen=True)
class Advice:
    category: str  # "Model" or "Augmentation"
    severity: str  # "info", "warning", "critical"
    finding: str
    suggestion: str


# Typical YOLO-family parameters.
DEFAULT_STRIDE = 32
TINY_FRACTION_WARN = 0.05  # >5 % tiny objects -> stride warning
EDGE_FRACTION_WARN = 0.30  # >30 % edge-touching -> random-crop warning
CENTERED_FRACTION_WARN = 0.7  # >70 % objects in middle third -> flip warning
IMBALANCE_RATIO_WARN = 10.0  # top/bottom class ratio -> mixup warning


def compute_advice(
    image_analysis: "ImageAnalysis",
    od_analysis: "ObjectDetectionAnalysis",
) -> List[Advice]:
    """Return actionable training-readiness advice based on the analyses."""
    advice: List[Advice] = []
    total_objs = od_analysis.total.num_objects

    # ---- Model: stride / resolution ----
    if total_objs > 0:
        tiny_frac = od_analysis.total.tiny_object_count / total_objs
        if tiny_frac > TINY_FRACTION_WARN:
            advice.append(Advice(
                category="Model",
                severity="warning",
                finding=(
                    f"{100 * tiny_frac:.0f} % of objects cover <0.5 % of image area. "
                    f"With default stride {DEFAULT_STRIDE} these are near or below "
                    "the feature-map resolution."
                ),
                suggestion=(
                    "Train at a larger input size (e.g. imgsz 1280), add a P2 "
                    "detection head, or use a smaller-stride backbone."
                ),
            ))

    # ---- Model: huge boxes = likely mislabels ----
    if od_analysis.total.huge_object_count > 0:
        advice.append(Advice(
            category="Model",
            severity="warning",
            finding=(
                f"{od_analysis.total.huge_object_count} object(s) cover > 50 % of "
                "their image. These usually indicate whole-scene mislabels."
            ),
            suggestion=(
                "Audit the huge-object list in `fix_first.csv` before training; "
                "full-image boxes poison the regression head."
            ),
        ))

    # ---- Model: anchor advice (only when we computed them) ----
    if od_analysis.recommended_anchors:
        ratios = [
            w / h for w, h in od_analysis.recommended_anchors if h > 0
        ]
        if ratios:
            min_r, max_r = min(ratios), max(ratios)
            if max_r > 3 * min_r:  # wide spread
                advice.append(Advice(
                    category="Model",
                    severity="info",
                    finding=(
                        f"Recommended anchor aspect ratios span {min_r:.2f}"
                        f"\u2013{max_r:.2f}."
                    ),
                    suggestion=(
                        "Replace default 1:1 / 1:2 / 2:1 anchors with the "
                        "priors from the 'Recommended anchor sizes' table; "
                        "default anchors will underperform on the extremes."
                    ),
                ))

    # ---- Augmentation: edge-touching rate -> random crop warning ----
    if total_objs > 0:
        edge_frac = od_analysis.total.edge_touching_count / total_objs
        if edge_frac > EDGE_FRACTION_WARN:
            advice.append(Advice(
                category="Augmentation",
                severity="warning",
                finding=(
                    f"{100 * edge_frac:.0f} % of objects already touch an image edge."
                ),
                suggestion=(
                    "Random-crop will destroy a lot of labels; prefer mosaic "
                    "(e.g. YOLO's built-in) or disable random-crop and use "
                    "letterbox + flip only."
                ),
            ))

    # ---- Augmentation: centered objects -> flip / crop advice ----
    hm = od_analysis.total.heatmap
    if hm is not None and hm.size > 0 and hm.sum() > 0:
        # The middle third (33-67 %) of both axes.
        n = hm.shape[0]
        lo = int(n / 3)
        hi = int(2 * n / 3)
        middle_sum = float(hm[lo:hi, lo:hi].sum())
        centered_frac = middle_sum / float(hm.sum())
        if centered_frac > CENTERED_FRACTION_WARN:
            advice.append(Advice(
                category="Augmentation",
                severity="info",
                finding=(
                    f"{100 * centered_frac:.0f} % of object mass lies in the "
                    "central third of the image."
                ),
                suggestion=(
                    "The spatial distribution is narrow; horizontal flip is safe, "
                    "but random-crop will often land in empty regions. Consider "
                    "center-crop-with-jitter augmentation."
                ),
            ))

    # ---- Augmentation: severe imbalance -> mixup warning ----
    counts = sorted(
        [c.num_objects for c in od_analysis.classes.values() if c.num_objects > 0]
    )
    if len(counts) >= 2 and counts[0] > 0:
        ratio = counts[-1] / counts[0]
        if ratio > IMBALANCE_RATIO_WARN:
            advice.append(Advice(
                category="Augmentation",
                severity="warning",
                finding=(
                    f"Most-common class has {ratio:.0f}\u00d7 more objects than "
                    "the rarest."
                ),
                suggestion=(
                    "Mixup and CutMix amplify this bias; prefer weighted sampling "
                    "(WeightedRandomSampler) and class-balanced loss over blend "
                    "augmentations."
                ),
            ))

    return advice
