"""Three curated dataset-overview plots for the findings-first report.

Each function renders one PNG into the caller's output folder and
returns the relative path (or empty string when the data is missing).

Kept separate from ``plots.py`` so this package stays scoped to the
check-based pipeline and doesn't pick up the legacy plot catalogue.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from lightly_insights.core.dataset import Dataset


def render_class_composition(output_folder: Path, dataset: Dataset) -> str:
    """Horizontal bar: one row per class, width = annotation count."""
    if not dataset.categories:
        return ""
    by_class = dataset.annotations_by_class
    rows = sorted(
        (
            (cat.name, len(by_class.get(cat.id, [])))
            for cat in dataset.categories
        ),
        key=lambda r: r[1],
    )
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    if sum(counts) == 0:
        return ""

    fig = plt.figure(figsize=(6, max(2, 0.35 * len(names) + 1)))
    ax = fig.add_subplot(111)
    bars = ax.barh(names, counts, color="#3498db", alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{c}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Annotations")
    ax.set_title("Class composition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = output_folder / "class_composition.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def render_confidence_distribution(
    output_folder: Path, dataset: Dataset
) -> str:
    """Overlaid histograms of confidence per class, when autolabels exist."""
    by_class = dataset.annotations_by_class
    series: List = []
    for cat in dataset.categories:
        confs = [
            a.confidence
            for a in by_class.get(cat.id, [])
            if a.confidence is not None
        ]
        if len(confs) < 10:
            continue  # too small to plot meaningfully
        series.append((cat.name, confs))
    if not series:
        return ""

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    bins = np.linspace(0.0, 1.0, 21)
    cmap = plt.get_cmap("tab10")
    for i, (name, confs) in enumerate(series):
        ax.hist(
            confs,
            bins=bins,
            alpha=0.45,
            label=name,
            color=cmap(i % 10),
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Annotations")
    ax.set_title("Autolabel confidence by class")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = output_folder / "confidence_distribution.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def render_spatial_heatmap(
    output_folder: Path, dataset: Dataset, grid: int = 50
) -> str:
    """2D density of where annotations land in normalized image coords.

    Uses tight_box for every geometry so it works for boxes, polygons,
    and masks. Normalized to 0-100 % on each axis.
    """
    if not dataset.annotations:
        return ""
    image_by_name = dataset.image_by_filename
    grid_array = np.zeros((grid, grid), dtype=np.float64)
    for ann in dataset.annotations:
        img = image_by_name.get(ann.image_filename)
        if img is None or img.width <= 0 or img.height <= 0:
            continue
        bb = ann.tight_box
        # Fractional coords.
        x1 = max(0.0, bb.xmin / img.width)
        x2 = min(1.0, bb.xmax / img.width)
        y1 = max(0.0, bb.ymin / img.height)
        y2 = min(1.0, bb.ymax / img.height)
        if x2 <= x1 or y2 <= y1:
            continue
        gx1, gx2 = int(x1 * grid), max(int(x1 * grid) + 1, int(x2 * grid))
        gy1, gy2 = int(y1 * grid), max(int(y1 * grid) + 1, int(y2 * grid))
        grid_array[gy1:gy2, gx1:gx2] += 1

    if grid_array.sum() == 0:
        return ""

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(grid_array, cmap="hot", extent=[0, 100, 100, 0], interpolation="nearest")
    ax.set_xlabel("X (%)")
    ax.set_ylabel("Y (%)")
    ax.set_title("Spatial density of annotations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = output_folder / "spatial_heatmap.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path.name
