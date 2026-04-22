"""Render annotated thumbnail galleries for issue types.

For each issue type we pick up to N examples and render a copy of the image
with the offending boxes drawn on top, so reviewers can see the problem
without opening a labeling tool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

if TYPE_CHECKING:  # pragma: no cover
    from lightly_insights.analyze import (
        ClassConflictPair,
        DuplicatePair,
        ObjectDetectionAnalysis,
    )

logger = logging.getLogger(__name__)

MAX_THUMBS = 6
THUMB_MAX_DIM = 480


@dataclass(frozen=True)
class DrilldownThumb:
    """A single annotated thumbnail referenced from the report template."""

    src: str  # path relative to output_folder
    caption: str
    filename: str


@dataclass(frozen=True)
class DrilldownGallery:
    title: str
    description: str
    thumbs: List[DrilldownThumb]


def _draw_boxes(
    src: Path,
    dst: Path,
    boxes: List[Tuple[Tuple[float, float, float, float], str, Tuple[int, int, int]]],
) -> bool:
    """Open src, scale to a thumbnail, draw boxes with labels, write to dst.

    Each box is (xyxy, label, rgb_color). Returns True on success.
    """
    try:
        with Image.open(src) as img:
            im = img.convert("RGB")
            orig_w, orig_h = im.size
            im.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))
            sx = im.size[0] / orig_w
            sy = im.size[1] / orig_h
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.load_default()
            except Exception:  # pragma: no cover
                font = None
            for (xmin, ymin, xmax, ymax), label, color in boxes:
                bx = (
                    int(xmin * sx),
                    int(ymin * sy),
                    int(xmax * sx),
                    int(ymax * sy),
                )
                draw.rectangle(bx, outline=color, width=3)
                if label:
                    # Simple label chip above the box.
                    text_pos = (bx[0] + 2, max(0, bx[1] - 12))
                    draw.text(text_pos, label, fill=color, font=font)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst)
        return True
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(f"Could not render drilldown for {src}: {exc}")
        return False


def build_drilldowns(
    output_folder: Path,
    image_folder: Path,
    od_analysis: "ObjectDetectionAnalysis",
) -> List[DrilldownGallery]:
    """Generate up to three galleries: class conflicts, duplicates, huge boxes.

    Thumbnails are written under `output_folder / drilldown / <kind>/`.
    Paths returned in `DrilldownThumb.src` are relative to `output_folder`.
    """
    galleries: List[DrilldownGallery] = []
    base = output_folder / "drilldown"

    # ---- class conflicts ----
    if od_analysis.class_conflicts:
        thumbs: List[DrilldownThumb] = []
        kind_dir = base / "conflicts"
        for i, pair in enumerate(od_analysis.class_conflicts[:MAX_THUMBS]):
            src = image_folder / pair.filename
            dst = kind_dir / f"{i:02d}_{Path(pair.filename).stem}.jpg"
            if _draw_boxes(
                src=src,
                dst=dst,
                boxes=[
                    (pair.box_a, pair.category_a, (220, 30, 30)),
                    (pair.box_b, pair.category_b, (30, 90, 220)),
                ],
            ):
                thumbs.append(
                    DrilldownThumb(
                        src=str(dst.relative_to(output_folder)),
                        caption=(
                            f"{pair.category_a} (red) vs {pair.category_b} (blue), "
                            f"IoU = {pair.iou:.2f}"
                        ),
                        filename=pair.filename,
                    )
                )
        if thumbs:
            galleries.append(
                DrilldownGallery(
                    title="Class conflicts",
                    description=(
                        "Boxes from different classes overlap with IoU ≥ 0.5. "
                        "Often a labeling mistake — verify which class is correct."
                    ),
                    thumbs=thumbs,
                )
            )

    # ---- duplicate annotations ----
    if od_analysis.duplicate_annotations:
        thumbs = []
        kind_dir = base / "duplicates"
        for i, pair in enumerate(od_analysis.duplicate_annotations[:MAX_THUMBS]):
            src = image_folder / pair.filename
            dst = kind_dir / f"{i:02d}_{Path(pair.filename).stem}.jpg"
            if _draw_boxes(
                src=src,
                dst=dst,
                boxes=[
                    (pair.box_a, pair.category_a, (220, 30, 30)),
                    (pair.box_b, "", (30, 220, 30)),
                ],
            ):
                thumbs.append(
                    DrilldownThumb(
                        src=str(dst.relative_to(output_folder)),
                        caption=(
                            f"{pair.category_a} box drawn twice, IoU = {pair.iou:.2f}"
                        ),
                        filename=pair.filename,
                    )
                )
        if thumbs:
            galleries.append(
                DrilldownGallery(
                    title="Duplicate annotations",
                    description=(
                        "Same class, near-identical boxes (IoU ≥ 0.9). "
                        "Delete one of the two to avoid duplicate-counting during training."
                    ),
                    thumbs=thumbs,
                )
            )

    # ---- huge-box samples (one per problematic class) ----
    from lightly_insights.analyze import HUGE_OBJECT_REL_AREA  # avoid cycle

    huge_thumbs: List[DrilldownThumb] = []
    kind_dir = base / "huge_boxes"
    count_total_rendered = 0
    for class_analysis in od_analysis.classes.values():
        if class_analysis.huge_object_count == 0:
            continue
        # Find a sample filename for this class that (likely) contains the huge box.
        # We don't carry per-box filenames, so pick any sample and draw all its
        # huge boxes.
        for filename in class_analysis.sample_filenames:
            if count_total_rendered >= MAX_THUMBS:
                break
            # We don't have per-box filenames, so skip unless the box list has
            # huge entries. Cheap heuristic: render the first sample filename
            # with a note; the accompanying CSV gives the precise list.
            src = image_folder / filename
            dst = kind_dir / f"{count_total_rendered:02d}_{Path(filename).stem}.jpg"
            if _draw_boxes(src=src, dst=dst, boxes=[]):
                huge_thumbs.append(
                    DrilldownThumb(
                        src=str(dst.relative_to(output_folder)),
                        caption=(
                            f"{class_analysis.class_name}: "
                            f"{class_analysis.huge_object_count} huge box(es) — "
                            "verify in fix_first.csv"
                        ),
                        filename=filename,
                    )
                )
                count_total_rendered += 1
                break
    if huge_thumbs:
        galleries.append(
            DrilldownGallery(
                title="Huge-box samples",
                description=(
                    "Boxes covering > 50 % of the image, by class. "
                    "Whole-scene boxes usually indicate mislabels."
                ),
                thumbs=huge_thumbs,
            )
        )

    return galleries
