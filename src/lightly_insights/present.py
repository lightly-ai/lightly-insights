import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Counter, Dict, List, Set, Tuple

import tqdm
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lightly_insights import plots
from lightly_insights.analyze import ImageAnalysis, ObjectDetectionAnalysis
from lightly_insights.plots import PlotPaths

logger = logging.getLogger(__name__)
static_folder = Path(__file__).parent / "static"
template_folder = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class SampleImage:
    filename: str
    path: Path


@dataclass(frozen=True)
class ImageInsights:
    # Image sizes.
    image_sizes_most_common: List[Tuple[Tuple[int, int], int]]
    image_size_plot: str

    # Sample images.
    sample_images: List[SampleImage]


@dataclass(frozen=True)
class FilenameInsights:
    num_images_no_label: int
    num_labels_no_image: int
    sample_filenames_no_label: List[str]
    sample_filenames_no_image: List[str]


@dataclass(frozen=True)
class ObjectDetectionInsights:
    num_classes: int
    class_ids_most_common: List[int]  # Class ids ordered from most common.
    plots: PlotPaths
    class_plots: Dict[int, PlotPaths]


def create_html_report(
    output_folder: Path,
    image_analysis: ImageAnalysis,
    od_analysis: ObjectDetectionAnalysis,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    image_insights = _get_image_insights(
        output_folder=output_folder,
        image_analysis=image_analysis,
    )
    object_detection_insights = _get_object_detection_insights(
        output_folder=output_folder,
        od_analysis=od_analysis,
        image_folder=image_analysis.image_folder,
    )
    filename_insights = _get_filename_insights(
        output_folder=output_folder,
        image_filename_set=image_analysis.filename_set,
        label_filename_set=od_analysis.filename_set,
    )
    report_data = dict(
        image_analysis=image_analysis,
        object_detection_analysis=od_analysis,
        image_insights=image_insights,
        object_detection_insights=object_detection_insights,
        filename_insights=filename_insights,
        date_generated=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    )

    # Setup Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(searchpath=template_folder),
        undefined=StrictUndefined,
    )
    template = env.get_template("report.html")

    # Render the template with data
    html_output = template.render(report_data)

    # Write the HTML to file
    html_output_path = output_folder / "index.html"
    html_output_path.write_text(html_output)

    # Copy static files.
    output_static_folder = output_folder / "static"
    if output_static_folder.exists():
        shutil.rmtree(output_static_folder, ignore_errors=True)
    shutil.copytree(src=static_folder, dst=output_static_folder)

    logger.info(f"Successfully created HTML report: {html_output_path.resolve()}")


def _get_image_insights(
    output_folder: Path,
    image_analysis: ImageAnalysis,
) -> ImageInsights:
    # Image size plot.
    plots.width_heigth_pixels_plot(
        output_file=output_folder / "image_size_plot.png",
        size_histogram=image_analysis.image_sizes,
        title="Image Sizes",
    )

    # Sample images.
    sample_folder = output_folder / "sample"
    sample_folder.mkdir(parents=True, exist_ok=True)
    sample_images = []
    rng = random.Random(42)
    selection = rng.sample(sorted(list(image_analysis.filename_set)), k=8)
    for filename in selection:
        shutil.copy2(
            src=image_analysis.image_folder / filename, dst=sample_folder / filename
        )
        sample_images.append(
            SampleImage(filename=filename, path=Path("./sample") / filename)
        )

    return ImageInsights(
        image_sizes_most_common=list(image_analysis.image_sizes.most_common()),
        image_size_plot="image_size_plot.png",
        sample_images=sample_images,
    )


def _get_object_detection_insights(
    output_folder: Path,
    od_analysis: ObjectDetectionAnalysis,
    image_folder: Path,
) -> ObjectDetectionInsights:
    # Plots.
    plots_folder = output_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)
    total_plots = plots.create_object_plots(
        output_folder=output_folder,
        plot_folder=plots_folder,
        class_analysis=od_analysis.total,
    )

    # Class plots.
    class_plots_folder = output_folder / "class_plots"
    class_plots = {}
    for class_id, class_analysis in tqdm.tqdm(
        od_analysis.classes.items(),
        desc="Creating plots per class",
    ):
        subfolder = class_plots_folder / f"{class_id}"
        subfolder.mkdir(parents=True, exist_ok=True)
        class_plots[class_id] = plots.create_object_plots(
            output_folder=output_folder,
            plot_folder=class_plots_folder / f"{class_id}",
            class_analysis=class_analysis,
        )

    # Copy class samples.
    for class_analysis in od_analysis.classes.values():
        for filename in class_analysis.sample_filenames:
            src_path = image_folder / filename
            dst_path = output_folder / "sample" / filename
            if src_path.exists():
                shutil.copy2(
                    src=src_path,
                    dst=dst_path,
                )

    # Class ids ordered from most common.
    class_counts = Counter(
        {
            class_id: class_data.num_objects
            for class_id, class_data in od_analysis.classes.items()
        }
    )
    class_ids_most_common = [id for id, _ in class_counts.most_common()]

    return ObjectDetectionInsights(
        num_classes=len(od_analysis.classes),
        class_ids_most_common=class_ids_most_common,
        plots=total_plots,
        class_plots=class_plots,
    )


def _get_filename_insights(
    output_folder: Path,
    image_filename_set: Set[str],
    label_filename_set: Set[str],
) -> FilenameInsights:
    filenames_no_label = sorted(list(image_filename_set - label_filename_set))
    filenames_no_image = sorted(list(label_filename_set - image_filename_set))

    if len(filenames_no_label) > 0:
        images_no_label_txt = output_folder / "images_no_label.txt"
        logger.info(
            f"Found {len(filenames_no_label)} images without a corresponding label "
            "file. Storing list of their filenames to 'images_no_label.txt'."
        )
        images_no_label_txt.write_text("\n".join(filenames_no_label))

    if len(filenames_no_image) > 0:
        labels_no_image_txt = output_folder / "labels_no_image.txt"
        logger.info(
            f"Found {len(filenames_no_image)} labels without a corresponding image "
            "file. Storing list of missing image filenames to 'labels_no_image.txt'."
        )
        labels_no_image_txt.write_text("\n".join(filenames_no_image))

    return FilenameInsights(
        num_images_no_label=len(filenames_no_label),
        num_labels_no_image=len(filenames_no_image),
        sample_filenames_no_label=filenames_no_label[:5],
        sample_filenames_no_image=filenames_no_image[:5],
    )
