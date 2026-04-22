"""Compare two previously-generated insights snapshots.

Consumes the `insights.json` that `present.create_html_report` writes and
produces a markdown diff. Designed for:
  - train/val/test drift checks
  - before/after re-labeling comparisons
  - dataset v1 vs. v2 tracking across commits
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Delta:
    label: str
    left: Any
    right: Any
    delta_str: str


def _delta_num(a: Optional[float], b: Optional[float], as_int: bool = False) -> str:
    """Render a delta like "+3 (+12%)" or "-0.05" for two numeric values."""
    if a is None or b is None:
        return "—"
    diff = b - a
    sign = "+" if diff >= 0 else ""
    if as_int:
        if a == 0:
            return f"{sign}{int(round(diff))}"
        pct = 100 * diff / a if a else 0
        return f"{sign}{int(round(diff))} ({sign}{pct:.0f}%)"
    return f"{sign}{diff:.2f}"


def load_snapshot(folder: Path) -> Dict[str, Any]:
    path = folder / "insights.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No insights.json in {folder}. Run lightly-insights on this folder first."
        )
    with path.open() as f:
        data: Dict[str, Any] = json.load(f)
    return data


def compare_snapshots(
    a: Dict[str, Any],
    b: Dict[str, Any],
    a_label: str = "A",
    b_label: str = "B",
) -> str:
    """Produce a markdown diff comparing two snapshots."""
    lines: List[str] = []
    lines.append(f"# Dataset comparison: {a_label} → {b_label}")
    lines.append("")

    # Top-line score.
    lines.append("## Health score")
    lines.append("")
    lines.append(f"| Metric | {a_label} | {b_label} | Δ |")
    lines.append("|---|---|---|---|")
    a_score = a.get("overall_score")
    b_score = b.get("overall_score")
    lines.append(
        f"| Overall score | "
        f"{'—' if a_score is None else f'{a_score:.0f}'} | "
        f"{'—' if b_score is None else f'{b_score:.0f}'} | "
        f"{_delta_num(a_score, b_score)} |"
    )
    lines.append(
        f"| Grade | {a.get('grade', '—')} | {b.get('grade', '—')} | — |"
    )
    lines.append("")

    # Subscores.
    lines.append("## Subscores")
    lines.append("")
    lines.append(f"| Subscore | {a_label} | {b_label} | Δ |")
    lines.append("|---|---|---|---|")
    a_subs = {s["name"]: s for s in a.get("subscores", [])}
    b_subs = {s["name"]: s for s in b.get("subscores", [])}
    for name in sorted(set(a_subs) | set(b_subs)):
        av = a_subs.get(name, {}).get("score")
        bv = b_subs.get(name, {}).get("score")
        av_str = "—" if av is None or av < 0 else f"{av:.0f}"
        bv_str = "—" if bv is None or bv < 0 else f"{bv:.0f}"
        lines.append(
            f"| {name} | {av_str} | {bv_str} | "
            f"{_delta_num(av if (av is not None and av >= 0) else None, bv if (bv is not None and bv >= 0) else None)} |"
        )
    lines.append("")

    # Counts.
    lines.append("## Counts")
    lines.append("")
    lines.append(f"| Metric | {a_label} | {b_label} | Δ |")
    lines.append("|---|---|---|---|")
    for key, label in [
        ("num_images", "Images"),
        ("num_objects", "Objects"),
        ("num_classes", "Classes"),
        ("corrupt_count", "Corrupt images"),
        ("class_conflicts_count", "Class conflicts"),
        ("duplicate_annotations_count", "Duplicate annotations"),
        ("near_duplicate_groups_count", "Near-duplicate image groups"),
    ]:
        av = a.get(key, 0)
        bv = b.get(key, 0)
        lines.append(
            f"| {label} | {av} | {bv} | {_delta_num(av, bv, as_int=True)} |"
        )
    lines.append("")

    # Changes in top issues (set diff).
    a_issues = set(a.get("issues", []))
    b_issues = set(b.get("issues", []))
    resolved = a_issues - b_issues
    introduced = b_issues - a_issues
    if resolved or introduced:
        lines.append("## Issue diff")
        lines.append("")
        if resolved:
            lines.append("**Resolved:**")
            for issue in sorted(resolved):
                lines.append(f"- ✅ {issue}")
            lines.append("")
        if introduced:
            lines.append("**New:**")
            for issue in sorted(introduced):
                lines.append(f"- ⚠️ {issue}")
            lines.append("")

    return "\n".join(lines) + "\n"


def write_comparison(
    a_folder: Path,
    b_folder: Path,
    output_file: Path,
    a_label: Optional[str] = None,
    b_label: Optional[str] = None,
) -> Path:
    """Read two insights.json files and write a markdown diff to output_file."""
    a = load_snapshot(a_folder)
    b = load_snapshot(b_folder)
    md = compare_snapshots(
        a=a,
        b=b,
        a_label=a_label or a_folder.name,
        b_label=b_label or b_folder.name,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(md)
    return output_file
