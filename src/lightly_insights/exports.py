"""Export helpers: CSV / JSON / Markdown worklists alongside the HTML report.

These are written by `create_html_report` into the same output folder so
labelers and CI can consume them without scraping HTML.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from lightly_insights.analyze import ImageAnalysis, ObjectDetectionAnalysis
    from lightly_insights.present import HealthScore


# Severity rubric: lower = fix first. The report lists issues in this order
# too so the CSV and HTML agree.
SEVERITY = {
    "leakage": 0,
    "corrupt": 10,
    "class_conflict": 20,
    "duplicate_annotation": 30,
    "starved_class": 40,
    "huge_object": 50,
    "near_duplicate": 60,
    "blurry": 70,
    "uniform": 75,
    "extreme_aspect": 80,
    "tiny_object": 90,
}


def export_worklists(
    output_folder: Path,
    image_analysis: "ImageAnalysis",
    od_analysis: "ObjectDetectionAnalysis",
    health_score: "HealthScore",
) -> List[Path]:
    """Write per-issue CSVs plus a single ranked fix_first.csv.

    Returns the list of paths written (for logging / verification).
    """
    written: List[Path] = []
    rows: List[Dict[str, Any]] = []

    # ---- corrupt files ----
    if image_analysis.corrupt_files:
        p = output_folder / "corrupt_files.txt"
        p.write_text("\n".join(image_analysis.corrupt_files) + "\n")
        written.append(p)
        for name in image_analysis.corrupt_files:
            rows.append(
                {
                    "severity": SEVERITY["corrupt"],
                    "issue_type": "corrupt",
                    "filename": name,
                    "detail": "Could not be read by PIL",
                    "action": "Remove or replace the file",
                }
            )

    # ---- class conflicts ----
    if od_analysis.class_conflicts:
        p = output_folder / "class_conflicts.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "filename", "category_a", "category_b", "iou",
                "box_a_xmin", "box_a_ymin", "box_a_xmax", "box_a_ymax",
                "box_b_xmin", "box_b_ymin", "box_b_xmax", "box_b_ymax",
            ])
            for pair in od_analysis.class_conflicts:
                w.writerow([
                    pair.filename, pair.category_a, pair.category_b,
                    f"{pair.iou:.3f}", *pair.box_a, *pair.box_b,
                ])
        written.append(p)
        for pair in od_analysis.class_conflicts:
            rows.append(
                {
                    "severity": SEVERITY["class_conflict"],
                    "issue_type": "class_conflict",
                    "filename": pair.filename,
                    "detail": (
                        f"{pair.category_a} vs {pair.category_b}, "
                        f"IoU={pair.iou:.2f}"
                    ),
                    "action": "Verify which class is correct; remove the wrong box",
                }
            )

    # ---- duplicate annotations ----
    if od_analysis.duplicate_annotations:
        p = output_folder / "duplicate_annotations.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "filename", "category_a", "category_b", "iou",
                "box_a_xmin", "box_a_ymin", "box_a_xmax", "box_a_ymax",
                "box_b_xmin", "box_b_ymin", "box_b_xmax", "box_b_ymax",
            ])
            for pair in od_analysis.duplicate_annotations:
                w.writerow([
                    pair.filename, pair.category_a, pair.category_b,
                    f"{pair.iou:.3f}", *pair.box_a, *pair.box_b,
                ])
        written.append(p)
        for pair in od_analysis.duplicate_annotations:
            rows.append(
                {
                    "severity": SEVERITY["duplicate_annotation"],
                    "issue_type": "duplicate_annotation",
                    "filename": pair.filename,
                    "detail": f"{pair.category_a} IoU={pair.iou:.2f}",
                    "action": "Delete one of the two boxes",
                }
            )

    # ---- near-duplicate image groups ----
    if image_analysis.near_duplicate_groups:
        p = output_folder / "near_duplicate_groups.json"
        p.write_text(
            json.dumps(image_analysis.near_duplicate_groups, indent=2) + "\n"
        )
        written.append(p)
        for group in image_analysis.near_duplicate_groups:
            for name in group[1:]:  # keep the first, flag the rest
                rows.append(
                    {
                        "severity": SEVERITY["near_duplicate"],
                        "issue_type": "near_duplicate",
                        "filename": name,
                        "detail": f"Near-duplicate of {group[0]}",
                        "action": "Dedupe the group",
                    }
                )

    # ---- quality flags (blur / uniform / extreme aspect) ----
    qf = image_analysis.quality_flags
    if qf is not None:
        for name in qf.blurry_files:
            rows.append({
                "severity": SEVERITY["blurry"],
                "issue_type": "blurry",
                "filename": name,
                "detail": "Laplacian variance below threshold",
                "action": "Inspect; remove if out of focus",
            })
        for name in qf.uniform_files:
            rows.append({
                "severity": SEVERITY["uniform"],
                "issue_type": "uniform",
                "filename": name,
                "detail": "Luminance std below threshold (all-black / all-white)",
                "action": "Inspect; likely corrupt download",
            })
        for name in qf.extreme_aspect_files:
            rows.append({
                "severity": SEVERITY["extreme_aspect"],
                "issue_type": "extreme_aspect",
                "filename": name,
                "detail": "Aspect ratio > 5:1 or < 1:5",
                "action": "Crop or remove if not intentional",
            })

    # ---- starved classes ----
    from lightly_insights.present import MIN_OBJECTS_PER_CLASS  # avoid cycle

    for cls in od_analysis.classes.values():
        if 0 < cls.num_objects < MIN_OBJECTS_PER_CLASS:
            rows.append({
                "severity": SEVERITY["starved_class"],
                "issue_type": "starved_class",
                "filename": "",
                "detail": (
                    f"Class '{cls.class_name}' has {cls.num_objects} objects "
                    f"(< {MIN_OBJECTS_PER_CLASS})"
                ),
                "action": "Collect more or merge into a parent class",
            })

    # ---- fix_first.csv (master list) ----
    if rows:
        rows.sort(key=lambda r: (r["severity"], r["issue_type"], r["filename"]))
        p = output_folder / "fix_first.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["severity", "issue_type", "filename", "detail", "action"]
            )
            w.writeheader()
            w.writerows(rows)
        written.append(p)

    # ---- machine-readable insights snapshot ----
    snapshot: Dict[str, Any] = {
        "overall_score": health_score.overall,
        "grade": health_score.grade,
        "subscores": [asdict(s) for s in health_score.subscores],
        "issues": list(health_score.issues),
        "num_images": image_analysis.num_images,
        "num_objects": od_analysis.total.num_objects,
        "num_classes": len(od_analysis.classes),
        "corrupt_count": len(image_analysis.corrupt_files),
        "class_conflicts_count": len(od_analysis.class_conflicts),
        "duplicate_annotations_count": len(od_analysis.duplicate_annotations),
        "near_duplicate_groups_count": len(image_analysis.near_duplicate_groups),
        "recommended_anchors": list(od_analysis.recommended_anchors),
    }
    p = output_folder / "insights.json"
    p.write_text(json.dumps(snapshot, indent=2) + "\n")
    written.append(p)

    return written


def export_markdown_summary(
    output_folder: Path,
    image_analysis: "ImageAnalysis",
    od_analysis: "ObjectDetectionAnalysis",
    health_score: "HealthScore",
) -> Path:
    """Executive summary users can paste into PRs / Slack / Notion."""
    lines: List[str] = []
    lines.append("# Dataset Insights Summary")
    lines.append("")
    if health_score.overall is not None:
        lines.append(
            f"**Health score: {health_score.overall:.0f}/100  "
            f"(Grade {health_score.grade})**"
        )
    lines.append("")
    lines.append(f"- Images: **{image_analysis.num_images}**")
    lines.append(f"- Objects: **{od_analysis.total.num_objects}**")
    lines.append(f"- Classes: **{len(od_analysis.classes)}**")
    lines.append("")

    if health_score.subscores:
        lines.append("## Subscores")
        lines.append("")
        lines.append("| Category | Score | Grade | Detail |")
        lines.append("|---|---|---|---|")
        for sub in health_score.subscores:
            if sub.score < 0:
                lines.append(f"| {sub.name} | — | — | {sub.detail} |")
            else:
                lines.append(
                    f"| {sub.name} | {sub.score:.0f} | {sub.grade} | {sub.detail} |"
                )
        lines.append("")

    if health_score.issues:
        lines.append("## Top issues to fix")
        lines.append("")
        for issue in health_score.issues:
            lines.append(f"- {issue}")
        lines.append("")

    if od_analysis.recommended_anchors:
        lines.append("## Recommended anchor priors")
        lines.append("")
        lines.append("| # | Width × Height | Aspect |")
        lines.append("|---|---|---|")
        for i, (w, h) in enumerate(od_analysis.recommended_anchors, 1):
            ratio = f"{w / h:.2f}" if h > 0 else "—"
            lines.append(f"| {i} | {w:.0f} × {h:.0f} | {ratio} |")
        lines.append("")

    lines.append(f"See the full HTML report at `{(output_folder / 'index.html').name}`.")
    lines.append("")

    p = output_folder / "SUMMARY.md"
    p.write_text("\n".join(lines))
    return p
