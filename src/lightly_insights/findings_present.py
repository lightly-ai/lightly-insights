"""Findings-first HTML report.

Consumes ``Dataset``, ``List[Finding]``, and ``List[ReviewItem]`` directly.
Does NOT rely on the legacy ``ImageAnalysis`` / ``ObjectDetectionAnalysis``
objects — this is the forward-facing renderer for LightlyStudio and
anyone else who produces findings through the check framework.

Writes:
- ``index.html`` — slim findings-first view
- ``review_queue.csv`` — from the review queue
- ``findings.json`` — machine-readable Finding dump
- ``static/`` — copied Bootstrap assets

The legacy ``present.create_html_report`` is unchanged; callers pick
which renderer to use.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity
from lightly_insights.core.overview_plots import (
    render_class_composition,
    render_confidence_distribution,
    render_cumulative_review_burden,
    render_findings_by_check,
    render_spatial_heatmap,
)
from lightly_insights.core.review_queue import (
    ReviewItem,
    export_review_queue_csv,
)

logger = logging.getLogger(__name__)
_static_folder = Path(__file__).parent / "static"
_template_folder = Path(__file__).parent / "templates"


def create_findings_report(
    output_folder: Path,
    dataset: Dataset,
    findings: Sequence[Finding],
    review_queue: Sequence[ReviewItem],
) -> Path:
    """Render the slim findings-first HTML report into ``output_folder``.

    Returns the path to ``index.html``.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Bucket severities for the summary strip.
    buckets = _severity_buckets(findings)

    # Group findings by category for the main list.
    by_cat: dict = defaultdict(list)
    for f in findings:
        by_cat[f.category].append(f)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda f: f.severity)
    findings_by_category = sorted(
        by_cat.items(),
        key=lambda kv: (min(f.severity for f in kv[1]), kv[0]),
    )

    # Sources in use, for the header.
    sources = sorted(
        {a.source for a in dataset.annotations if a.source is not None}
    )
    sources_joined = ", ".join(sources) if sources else "unknown"

    # CSV + JSON side-car exports.
    review_queue_csv_path = output_folder / "review_queue.csv"
    export_review_queue_csv(review_queue, review_queue_csv_path)
    (output_folder / "findings.json").write_text(
        json.dumps([f.to_dict() for f in findings], indent=2) + "\n"
    )

    # Static assets (Bootstrap etc.) — same folder legacy uses.
    output_static = output_folder / "static"
    if not output_static.exists():
        shutil.copytree(src=_static_folder, dst=output_static)

    # Overview plots. Each returns "" if the underlying data isn't there.
    class_plot = render_class_composition(output_folder, dataset)
    confidence_plot = render_confidence_distribution(output_folder, dataset)
    spatial_plot = render_spatial_heatmap(output_folder, dataset)
    by_check_plot = render_findings_by_check(output_folder, findings)
    burden_plot = render_cumulative_review_burden(output_folder, review_queue)

    env = Environment(
        loader=FileSystemLoader(searchpath=_template_folder),
        undefined=StrictUndefined,
    )
    template = env.get_template("findings_report.html")
    html = template.render(
        dataset=dataset,
        findings=list(findings),
        findings_by_category=findings_by_category,
        review_queue=list(review_queue),
        review_queue_path="review_queue.csv",
        severity_buckets=buckets,
        sources_joined=sources_joined,
        overview_plots={
            "class_composition": class_plot,
            "confidence_distribution": confidence_plot,
            "spatial_heatmap": spatial_plot,
            "findings_by_check": by_check_plot,
            "review_burden": burden_plot,
        },
        date_generated=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    )
    index_path = output_folder / "index.html"
    index_path.write_text(html)

    logger.info(f"Findings report written to {index_path.resolve()}")
    return index_path


def _severity_buckets(findings: Iterable[Finding]) -> List:
    """Return (label, count, color) triples for the summary strip."""
    ranges = [
        ("Critical", (Severity.CRITICAL, 20), "#c0392b"),
        ("High", (Severity.HIGH, 40), "#e67e22"),
        ("Medium", (Severity.MEDIUM, 60), "#f1c40f"),
        ("Info/Low", (Severity.LOW, 100), "#7f8c8d"),
    ]
    buckets: List = []
    for label, (lo, hi), color in ranges:
        count = sum(1 for f in findings if lo <= f.severity < hi)
        buckets.append((label, count, color))
    return buckets
