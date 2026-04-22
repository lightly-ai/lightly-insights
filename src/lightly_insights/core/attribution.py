"""Aggregate findings by annotation source.

When annotations carry ``source`` (``"human"``, ``"yolo-v11"``, model
version, or labeler id), per-source aggregation answers questions no
single finding can: "Which autolabeler is producing the most problems?"
"Is labeler #7 the root cause of half of my class conflicts?"

The output is a ranked list that feeds into both an HTML section and a
plot. Sorted by ``findings_per_100`` descending so the worst offender
is at the top.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding


@dataclass(frozen=True)
class AttributionItem:
    """Findings attributed to one annotation source."""

    source: str
    num_annotations: int
    num_findings: int  # count of findings touching this source
    findings_per_100: float  # rate per 100 annotations
    top_check_ids: List[Tuple[str, int]]  # the 3 most-common check_ids


def compute_attribution(
    dataset: Dataset,
    findings: Sequence[Finding],
) -> List[AttributionItem]:
    """Bucket findings by the ``source`` of the annotations they touch.

    A finding touching N distinct sources attributes once to each of
    them. Sources with zero annotations are skipped (they can't really
    have a rate). Sources with no findings still appear as a row so
    reviewers see who's clean.
    """
    source_counts: Counter = Counter()
    for ann in dataset.annotations:
        if ann.source is not None:
            source_counts[ann.source] += 1
    if not source_counts:
        return []

    # ann_id -> source, for quick lookup.
    source_by_ann: Dict[int, str] = {
        a.annotation_id: a.source
        for a in dataset.annotations
        if a.source is not None
    }

    findings_per_source: Counter = Counter()
    checks_per_source: Dict[str, Counter] = defaultdict(Counter)
    for f in findings:
        # Find every distinct source this finding references.
        sources_touched: set = set()
        for ann_id in f.affected_annotations:
            src = source_by_ann.get(ann_id)
            if src is not None:
                sources_touched.add(src)
        for src in sources_touched:
            findings_per_source[src] += 1
            checks_per_source[src][f.check_id] += 1

    items: List[AttributionItem] = []
    for source, num_ann in source_counts.items():
        nf = findings_per_source.get(source, 0)
        rate = (100.0 * nf / num_ann) if num_ann > 0 else 0.0
        top = checks_per_source[source].most_common(3)
        items.append(
            AttributionItem(
                source=source,
                num_annotations=num_ann,
                num_findings=nf,
                findings_per_100=round(rate, 2),
                top_check_ids=top,
            )
        )
    # Worst offender first; ties broken by absolute count then name.
    items.sort(
        key=lambda it: (-it.findings_per_100, -it.num_findings, it.source)
    )
    return items


def render_attribution_plot(
    output_folder: Path, items: Sequence[AttributionItem]
) -> str:
    """Horizontal bar: findings per 100 annotations, one row per source."""
    if not items:
        return ""
    import matplotlib.pyplot as plt

    ordered = sorted(items, key=lambda it: it.findings_per_100)
    names = [it.source for it in ordered]
    values = [it.findings_per_100 for it in ordered]
    counts = [it.num_annotations for it in ordered]

    fig = plt.figure(figsize=(6, max(2, 0.35 * len(names) + 1)))
    ax = fig.add_subplot(111)
    bars = ax.barh(names, values, color="#8e44ad", alpha=0.85)
    max_v = max(values) if values else 1
    for bar, v, n in zip(bars, values, counts):
        ax.text(
            bar.get_width() + max_v * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}  (n={n})",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Findings per 100 annotations")
    ax.set_title("Findings by source")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = output_folder / "attribution.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path.name
