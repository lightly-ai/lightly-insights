"""Per-finding drill-down thumbnails for the findings-first report.

For each review-queue item that points at a specific annotation, render
a side-by-side pair:

- The flagged image with the offending box(es) drawn in red.
- A class exemplar — a representative annotation of the same class
  that no finding flagged — drawn in green. Lets reviewers compare
  "what the model got wrong" to "what a good example looks like."

Exemplars are computed once per class and reused across all drill-downs
for that class, keeping the pass linear in (findings × images-touched).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image as PILImage
from PIL import ImageDraw, UnidentifiedImageError

from lightly_insights.core.dataset import Annotation, AnnotationKind, Dataset
from lightly_insights.core.finding import Finding
from lightly_insights.core.review_queue import ReviewItem

logger = logging.getLogger(__name__)

THUMB_MAX_DIM = 600
# Top-N items from the review queue we bother to render. Thumbnails are the
# expensive part of report generation; cap to keep wall-clock reasonable.
MAX_DRILLDOWNS = 30


@dataclass(frozen=True)
class DrilldownPanel:
    """A single reviewer panel: one flagged thumbnail + its class exemplar."""

    anchor: str  # HTML id, e.g. "finding-1"
    rank: int  # matches the review queue rank
    title: str
    detail: str
    action: str
    flagged_src: str  # path relative to output_folder
    flagged_caption: str
    exemplar_src: str  # "" if no exemplar available
    exemplar_caption: str


def build_drilldowns(
    output_folder: Path,
    dataset: Dataset,
    findings: Sequence[Finding],
    review_queue: Sequence[ReviewItem],
    max_items: int = MAX_DRILLDOWNS,
) -> List[DrilldownPanel]:
    """Generate drill-down thumbnails for the top ``max_items`` review-queue rows.

    Returns panels in the same order as the review queue. Panels whose
    image can't be read are skipped silently; the caller renders what comes
    back.
    """
    findings_by_id: Dict[str, Finding] = {}
    for f in findings:
        # Key by (check_id, first affected annotation) so multiple rows with
        # the same check but different anns don't collide.
        for ann_id in f.affected_annotations:
            findings_by_id[f"{f.check_id}:{ann_id}"] = f

    drill_dir = output_folder / "drilldown"
    drill_dir.mkdir(parents=True, exist_ok=True)

    # Precompute id -> Annotation so downstream lookups are O(1) even when
    # annotation ids aren't contiguous (e.g. ml-proposal ids start at 10M).
    ann_by_id: Dict[int, Annotation] = {
        a.annotation_id: a for a in dataset.annotations
    }
    exemplar_cache: Dict[int, Tuple[str, str]] = {}  # class_id -> (path, caption)

    panels: List[DrilldownPanel] = []
    for item in list(review_queue)[:max_items]:
        if item.annotation_id is None:
            continue
        ann = ann_by_id.get(item.annotation_id)
        if ann is None:
            continue

        image = dataset.image_by_filename.get(ann.image_filename)
        if image is None or image.path is None or not image.path.exists():
            continue

        flagged_path = drill_dir / f"flagged_{item.rank:03d}.jpg"
        ok = _draw_annotation(
            src=image.path,
            dst=flagged_path,
            ann=ann,
            color=(220, 50, 50),
        )
        if not ok:
            continue

        flagged_caption = _format_caption(ann, prefix="flagged")

        # Exemplar for this class, memoized.
        if ann.class_id not in exemplar_cache:
            exemplar_cache[ann.class_id] = _build_exemplar(
                class_id=ann.class_id,
                dataset=dataset,
                findings=findings,
                output_folder=drill_dir,
            )
        exemplar_src, exemplar_caption = exemplar_cache[ann.class_id]

        panels.append(
            DrilldownPanel(
                anchor=f"finding-{item.rank}",
                rank=item.rank,
                title=item.title,
                detail=_lookup_detail(item, findings_by_id),
                action=item.action,
                flagged_src=str(flagged_path.relative_to(output_folder)),
                flagged_caption=flagged_caption,
                exemplar_src=exemplar_src,
                exemplar_caption=exemplar_caption,
            )
        )
    return panels


def _lookup_detail(item: ReviewItem, findings_by_id: Dict[str, Finding]) -> str:
    if item.annotation_id is None:
        return ""
    f = findings_by_id.get(f"{item.finding_id}:{item.annotation_id}")
    return f.detail if f is not None else ""


def _format_caption(ann: Annotation, prefix: str) -> str:
    parts = [f"{prefix}: {ann.class_name}"]
    if ann.confidence is not None:
        parts.append(f"conf={ann.confidence:.2f}")
    if ann.source:
        parts.append(ann.source)
    parts.append(ann.image_filename)
    return " · ".join(parts)


def _build_exemplar(
    class_id: int,
    dataset: Dataset,
    findings: Sequence[Finding],
    output_folder: Path,
) -> Tuple[str, str]:
    """Pick a 'typical' annotation for a class that no finding flagged."""
    flagged_ids = {
        aid
        for f in findings
        for aid in f.affected_annotations
    }
    candidates = [
        a for a in dataset.annotations_by_class.get(class_id, [])
        if a.annotation_id not in flagged_ids
    ]
    if not candidates:
        return "", ""
    # Pick the one whose (area, aspect) is closest to the class median.
    areas = [a.area for a in candidates]
    ratios = [a.tight_box.aspect_ratio for a in candidates if a.tight_box.height > 0]
    if not areas:
        return "", ""
    med_area = median(areas)
    med_ratio = median(ratios) if ratios else 0.0
    def _dist(a: Annotation) -> float:
        ar = a.tight_box.aspect_ratio if a.tight_box.height > 0 else 0.0
        area_term = (a.area - med_area) ** 2 / (med_area ** 2 + 1)
        ratio_term = (ar - med_ratio) ** 2 / (med_ratio ** 2 + 1)
        return area_term + ratio_term
    exemplar = min(candidates, key=_dist)
    image = dataset.image_by_filename.get(exemplar.image_filename)
    if image is None or image.path is None or not image.path.exists():
        return "", ""
    dst = output_folder / f"exemplar_class{class_id}.jpg"
    if _draw_annotation(src=image.path, dst=dst, ann=exemplar, color=(30, 180, 60)):
        return (
            str(dst.relative_to(output_folder.parent)),
            _format_caption(exemplar, prefix="typical"),
        )
    return "", ""


def _draw_annotation(
    src: Path,
    dst: Path,
    ann: Annotation,
    color: Tuple[int, int, int],
) -> bool:
    """Write a thumbnail of ``src`` with ``ann`` drawn on top.

    Returns True on success. False (and logs) when the image can't be read
    or the geometry can't be drawn.
    """
    try:
        with PILImage.open(src) as img:
            im = img.convert("RGB")
            orig_w, orig_h = im.size
            im.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))
            sx = im.size[0] / orig_w if orig_w else 1.0
            sy = im.size[1] / orig_h if orig_h else 1.0
            draw = ImageDraw.Draw(im)
            bb = ann.tight_box
            box = (
                int(bb.xmin * sx), int(bb.ymin * sy),
                int(bb.xmax * sx), int(bb.ymax * sy),
            )
            draw.rectangle(box, outline=color, width=3)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst)
        return True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(f"Could not render drilldown for {src}: {exc}")
        return False
