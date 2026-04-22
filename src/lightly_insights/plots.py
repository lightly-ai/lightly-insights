from dataclasses import dataclass
from pathlib import Path
from typing import Counter, List, Tuple, Union

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
    aspect_ratio: str


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
    aspect_ratio_path = plot_folder / "aspect_ratio.png"

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

    # Side length histogram. Bucket by multiples of 50px.
    side_length_avg_histogram = Counter(
        50.0 * round(((w + h) / 2) / 50) for w, h in class_analysis.object_sizes_abs
    )
    _histogram(
        output_file=side_length_avg_path,
        hist=side_length_avg_histogram,
        title="Object Side Length Average (buckets by 50px)",
        xlabel="(Width + Height) / 2 (px)",
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

    # Aspect-ratio distribution (log-scale so 2:1 and 1:2 are symmetric).
    _aspect_ratio_plot(
        output_file=aspect_ratio_path,
        aspect_ratios=class_analysis.aspect_ratios,
    )

    return PlotPaths(
        object_sizes_abs=str(object_sizes_abs_path.relative_to(output_folder)),
        object_sizes_rel=str(object_sizes_rel_path.relative_to(output_folder)),
        side_length_avg=str(side_length_avg_path.relative_to(output_folder)),
        rel_area=str(rel_area_path.relative_to(output_folder)),
        objects_per_image=str(objects_per_image_path.relative_to(output_folder)),
        heatmap=str(heatmap_path.relative_to(output_folder)),
        aspect_ratio=str(aspect_ratio_path.relative_to(output_folder)),
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


def cooccurrence_plot(
    output_file: Path,
    matrix: NDArray[np.int_],
    class_names: List[str],
) -> None:
    """Render a co-occurrence heatmap with class-name tick labels."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    # Log-scale color so a few dominant pairs don't wash out the rest.
    # +1 keeps zero-valued cells visible as the lightest shade.
    display = np.log1p(matrix.astype(np.float64))
    im = ax.imshow(display, cmap="Blues", interpolation="nearest")

    # Annotate cells with the raw counts.
    n = matrix.shape[0]
    max_val = float(matrix.max()) if matrix.size else 0.0
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val == 0:
                continue
            color = "white" if display[i, j] > np.log1p(max_val) * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Class Co-occurrence (images containing both classes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1 + count)")
    fig.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


def _aspect_ratio_plot(
    output_file: Path,
    aspect_ratios: List[float],
) -> None:
    """Log-scaled histogram of w/h ratios.

    Using log-scale puts 2:1 and 1:2 at symmetric distances from 1:1, which
    matches how detector anchor tuning thinks about aspect ratios.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if aspect_ratios:
        # Clip to [0.1, 10] to keep degenerate ratios from dominating the axis.
        clipped = np.clip(aspect_ratios, 0.1, 10.0)
        log_ratios = np.log10(clipped)
        ax.hist(log_ratios, bins=40, color="blue", alpha=0.5)
        # Reference lines at 1:2, 1:1, 2:1.
        for ref_ratio, label in [(0.5, "1:2"), (1.0, "1:1"), (2.0, "2:1")]:
            ax.axvline(
                x=np.log10(ref_ratio),
                color="gray",
                linestyle=":",
                alpha=0.7,
            )
            ax.text(
                np.log10(ref_ratio),
                ax.get_ylim()[1] * 0.95,
                label,
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                color="gray",
                fontsize=8,
            )
        # Tick labels show real ratios, not log values.
        ticks = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel("Width / Height")
    ax.set_ylabel("Number of Objects")
    ax.set_title("Aspect Ratio Distribution (log scale)")

    plt.savefig(output_file)
    plt.close(fig)
