"""Build a prioritized review queue from findings + per-annotation confidence.

The ranking rule is simple and opinionated:

    priority = (severity_weight) * (uncertainty_weight)

where

    severity_weight = max(1, 100 - finding.severity)  # higher = worse
    uncertainty_weight = 1.0 + (1.0 - confidence)     # ≥1, extra for low conf

For findings that name specific annotations, we pull the confidence of
those annotations and boost the priority. For findings that don't name
any (dataset-level findings like ``background_scarcity``), the priority
is just the severity weight.

The result is a list of ``ReviewItem`` rows that can be CSV-exported or
consumed by LightlyStudio directly. Lower ``rank`` = look at first.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding


@dataclass(frozen=True)
class ReviewItem:
    rank: int  # 1-based, 1 = highest priority
    priority: float
    finding_id: str  # check_id
    severity: int
    filename: str  # "" when finding is dataset-level
    annotation_id: Optional[int]
    confidence: Optional[float]
    title: str
    action: str


def build_review_queue(
    findings: Sequence[Finding],
    dataset: Dataset,
    max_items: Optional[int] = None,
) -> List[ReviewItem]:
    """Rank findings + their affected annotations into a review queue."""
    ann_by_id: Dict[int, "object"] = {a.annotation_id: a for a in dataset.annotations}

    rows: List[ReviewItem] = []
    for f in findings:
        sev_w = max(1.0, 100.0 - f.severity)

        if not f.affected_annotations and not f.affected_images:
            # Dataset-level finding: one row, priority = severity only.
            rows.append(
                ReviewItem(
                    rank=0,  # filled after sorting
                    priority=sev_w,
                    finding_id=f.check_id,
                    severity=f.severity,
                    filename="",
                    annotation_id=None,
                    confidence=None,
                    title=f.title,
                    action=f.action,
                )
            )
            continue

        if f.affected_annotations:
            for ann_id in f.affected_annotations:
                ann = ann_by_id.get(ann_id)
                conf = getattr(ann, "confidence", None)
                filename = getattr(ann, "image_filename", "")
                unc = 1.0 + (1.0 - conf) if conf is not None else 1.0
                rows.append(
                    ReviewItem(
                        rank=0,
                        priority=sev_w * unc,
                        finding_id=f.check_id,
                        severity=f.severity,
                        filename=filename,
                        annotation_id=ann_id,
                        confidence=conf,
                        title=f.title,
                        action=f.action,
                    )
                )
        else:
            # Only image filenames, no specific annotation ids.
            for filename in f.affected_images:
                rows.append(
                    ReviewItem(
                        rank=0,
                        priority=sev_w,
                        finding_id=f.check_id,
                        severity=f.severity,
                        filename=filename,
                        annotation_id=None,
                        confidence=None,
                        title=f.title,
                        action=f.action,
                    )
                )

    # Higher priority = looked at first. Break ties by severity (lower=worse),
    # then by filename for determinism.
    rows.sort(key=lambda r: (-r.priority, r.severity, r.filename, r.annotation_id or 0))
    if max_items is not None:
        rows = rows[:max_items]
    # Assign ranks.
    return [
        ReviewItem(
            rank=i + 1,
            priority=r.priority,
            finding_id=r.finding_id,
            severity=r.severity,
            filename=r.filename,
            annotation_id=r.annotation_id,
            confidence=r.confidence,
            title=r.title,
            action=r.action,
        )
        for i, r in enumerate(rows)
    ]


def export_review_queue_csv(items: Sequence[ReviewItem], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "priority", "finding", "severity",
            "filename", "annotation_id", "confidence",
            "title", "action",
        ])
        for r in items:
            writer.writerow([
                r.rank,
                f"{r.priority:.2f}",
                r.finding_id,
                r.severity,
                r.filename,
                "" if r.annotation_id is None else r.annotation_id,
                "" if r.confidence is None else f"{r.confidence:.3f}",
                r.title,
                r.action,
            ])
    return output_path
