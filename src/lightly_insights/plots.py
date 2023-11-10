from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from lightly_insights.analyze import ClassAnalysis


@dataclass(frozen=True)
class PlotPaths:
    object_sizes_abs: str
    object_sizes_rel: str
    side_length_avg: str
    rel_area: str
    objects_per_image: str
    heatmap: str


def create_object_plots(
    output_folder: Path,
    plot_folder: Path,
    class_analysis: ClassAnalysis,
) -> PlotPaths:
    """Create plots for object detection analysis.

    Output folder must be a parent of the plot folder. Returns plot paths relative
    to the output folder.
    """
    object_sizes_abs_path = plot_folder / "object_sizes_abs.png"
    object_sizes_rel_path = plot_folder / "object_sizes_rel.png"
    side_length_avg_path = plot_folder / "side_length_avg_plot.png"
    rel_area_path = plot_folder / "rel_area.png"
    objects_per_image_path = plot_folder / "objects_per_image.png"
    heatmap_path = plot_folder / "heatmap.png"

    # Bucket by multiples of 20px.
    size_histogram_abs = Counter(
        [
            (20.0 * round(w / 20), 20.0 * round(h / 20))
            for w, h in class_analysis.object_sizes_abs
        ]
    )
    width_heigth_pixels_plot(
        output_file=object_sizes_abs_path,
        size_histogram=size_histogram_abs,
        title="Object Sizes in Pixels (buckets by 20px)",
    )

    # Bucket by multiples of 5%.
    size_histogram_rel = Counter(
        [
            (100 * 0.05 * round(w / 0.05), 100 * 0.05 * round(h / 0.05))
            for w, h in class_analysis.object_sizes_rel
        ]
    )
    _width_heigth_percent_plot(
        output_file=object_sizes_rel_path,
        size_histogram=size_histogram_rel,
        title="Object Sizes in Percent  (buckets by 5%)",
    )

    # Side length histogram. Bucket by multiples of 20px.
    side_length_avg_histogram = Counter(
        50.0 * round((w + h / 2) / 50) for w, h in class_analysis.object_sizes_abs
    )
    _histogram(
        output_file=side_length_avg_path,
        hist=side_length_avg_histogram,
        title="Object Side Length Average (buckets by 50px)",
        xlabel="Width/2 + Height/2 (px)",
        ylabel="Number of Objects",
        bar_width=50,
        x_average_line=True,
    )

    # Side length histogram. Bucket by multiples of 5%.
    rel_area_histogram = Counter(
        100 * 0.05 * round(w * h / 0.05) for w, h in class_analysis.object_sizes_rel
    )
    _histogram(
        output_file=rel_area_path,
        hist=rel_area_histogram,
        title="Object Relative Area (buckets by 5%)",
        xlabel="Object Area (% of Image Area)",
        ylabel="Number of Objects",
        bar_width=100 * 0.05,
        x_average_line=True,
    )

    # Objects per image.
    _histogram(
        output_file=objects_per_image_path,
        hist=class_analysis.objects_per_image,
        title="Objects per Image",
        xlabel="Number of Objects",
        ylabel="Number of Images",
        bar_width=1.0,
        y_average_line=True,
    )

    # Heatmap.
    _heatmap(
        output_file=heatmap_path,
        heatmap=class_analysis.heatmap,
    )

    return PlotPaths(
        object_sizes_abs=str(object_sizes_abs_path.relative_to(output_folder)),
        object_sizes_rel=str(object_sizes_rel_path.relative_to(output_folder)),
        side_length_avg=str(side_length_avg_path.relative_to(output_folder)),
        rel_area=str(rel_area_path.relative_to(output_folder)),
        objects_per_image=str(objects_per_image_path.relative_to(output_folder)),
        heatmap=str(heatmap_path.relative_to(output_folder)),
    )


def width_heigth_pixels_plot(
    output_file: Path,
    size_histogram: Union[Counter[Tuple[float, float]], Counter[Tuple[int, int]]],
    title: str,
) -> None:
    # Image size plot.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    xs = []
    ys = []
    sizes = []
    for size, count in size_histogram.items():
        xs.append(size[0])
        ys.append(size[1])
        sizes.append(count)
    ax.scatter(
        xs,
        ys,
        s=sizes,
        marker="o",
        color="blue",
        alpha=0.5,
    )
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    if len(xs) > 0 and len(ys) > 0:
        ax.set_xlim(0, max(xs) * 1.1)
        ax.set_ylim(0, max(ys) * 1.1)

    # Save the plot.
    plt.savefig(output_file)
    plt.close(fig)


def _width_heigth_percent_plot(
    output_file: Path,
    size_histogram: Counter[Tuple[float, float]],
    title: str,
) -> None:
    # Image size plot.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    xs = []
    ys = []
    sizes = []
    for size, count in size_histogram.items():
        xs.append(size[0])
        ys.append(size[1])
        sizes.append(count)
    ax.scatter(
        xs,
        ys,
        s=sizes,
        marker="o",
        color="blue",
        alpha=0.5,
    )
    ax.set_xlabel("Width (%)")
    ax.set_ylabel("Height (%)")
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Save the plot.
    plt.savefig(output_file)
    plt.close(fig)


def _histogram(
    output_file: Path,
    hist: Union[Counter[int], Counter[float]],
    bar_width: float,
    title: str,
    xlabel: str,
    ylabel: str,
    x_average_line: bool = False,
    y_average_line: bool = False,
) -> None:
    # Image size plot.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # Vertical bars.
    xs = []
    ys = []
    for bucket, count in hist.items():
        xs.append(bucket)
        ys.append(count)
    ax.bar(
        xs,
        ys,
        color="blue",
        alpha=0.5,
        width=bar_width * 0.85,
    )

    # Vertical line.
    if x_average_line and sum(ys) > 0:
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        avg = sum_xy / sum(ys)
        ax.axvline(
            x=avg,
            color="red",
            linestyle="--",
        )
        ax.text(
            0.95,
            0.95,
            f"avg={avg:.1f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color="red",
        )

    # Horizontal line.
    if y_average_line and len(ys) > 0:
        avg = sum(ys) / len(ys)
        ax.axhline(
            y=avg,
            color="red",
            linestyle="--",
        )
        ax.text(
            0.95,
            0.95,
            f"avg={avg:.1f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color="red",
        )

    # Show x-ticks only at integers.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save the plot.
    plt.savefig(output_file)
    plt.close(fig)


def _heatmap(
    output_file: Path,
    heatmap: NDArray[np.float_],
) -> None:
    # Image size plot.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.imshow(
        heatmap,
        cmap="cividis",
        # cmap="viridis",
        # cmap="BuGn",
        # cmap="hot",
        # cmap="hot",
        interpolation="nearest",
    )

    ax.set_xlabel("X (%)")
    ax.set_ylabel("Y (%)")
    ax.set_title("Object Location Heatmap")

    # Save the plot.
    plt.savefig(output_file)
    plt.close(fig)
