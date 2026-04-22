import logging
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

import tqdm
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from lightly_insights import plots
from lightly_insights.analyze import ImageAnalysis, ObjectDetectionAnalysis
from lightly_insights.plots import PlotPaths

# Classes with fewer than this fraction of total objects are flagged as
# under-represented (see ImbalanceStats).
UNDER_REPRESENTED_FRACTION = 0.01

# Health score thresholds (tune these if your domain differs).
MIN_OBJECTS_PER_CLASS = 30  # fewer than this = nearly unlearnable
IDEAL_OBJECTS_PER_CLASS = 500  # at or above this = healthy
# Weights (must sum to 1.0).
HEALTH_WEIGHTS = {
    "balance": 0.25,
    "per_class_data": 0.20,
    "annotation_quality": 0.25,
    "image_quality": 0.20,
    "leakage": 0.10,
}

logger = logging.getLogger(__name__)
static_folder = Path(__file__).parent / "static"
template_folder = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class SampleImage:
    filename: str
    path: Path
    label: str = ""  # short tag explaining why it was picked (e.g. "smallest")


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
class Subscore:
    """One slice of the health score: 0-100 plus a letter grade and reason."""

    name: str
    score: float  # 0-100, or -1 for "N/A" (not computable from available data)
    grade: str  # "A"-"F" or "—"
    detail: str  # human-readable explanation


@dataclass(frozen=True)
class HealthScore:
    """Opinionated single-number dataset health read.

    `overall` is None when the dataset doesn't have enough signal to score
    (e.g. images-only reports). Subscores that were computable are still
    populated in `subscores`.
    """

    overall: Optional[float]  # 0-100 or None
    grade: str  # "A"-"F" or "—"
    subscores: List[Subscore]
    issues: List[str]  # actionable bullets, highest-priority first


@dataclass(frozen=True)
class ImbalanceStats:
    entropy: float  # nats
    normalized_entropy: float  # 0..1 (1 = perfectly balanced)
    gini: float  # 0..1 (0 = perfectly balanced)
    top_class_share: float  # 0..1
    under_represented_count: int  # classes below UNDER_REPRESENTED_FRACTION


@dataclass(frozen=True)
class ObjectDetectionInsights:
    num_classes: int
    class_ids_most_common: List[int]  # Class ids ordered from most common.
    plots: PlotPaths
    class_plots: Dict[int, PlotPaths]
    avg_objects_per_image: float
    avg_objects_per_class: float
    avg_images_per_class: float
    imbalance: ImbalanceStats
    cooccurrence_plot: str = ""


def create_html_report(
    output_folder: Path,
    image_analysis: ImageAnalysis,
    od_analysis: ObjectDetectionAnalysis,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    image_insights = _get_image_insights(
        output_folder=output_folder,
        image_analysis=image_analysis,
        od_analysis=od_analysis,
    )
    object_detection_insights = _get_object_detection_insights(
        output_folder=output_folder,
        od_analysis=od_analysis,
        image_folder=image_analysis.image_folder,
        num_images=image_analysis.num_images,
    )
    filename_insights = _get_filename_insights(
        output_folder=output_folder,
        image_filename_set=image_analysis.filename_set,
        label_filename_set=od_analysis.filename_set,
    )
    health_score = compute_health_score(
        image_analysis=image_analysis, od_analysis=od_analysis
    )

    report_data = dict(
        image_analysis=image_analysis,
        object_detection_analysis=od_analysis,
        image_insights=image_insights,
        object_detection_insights=object_detection_insights,
        filename_insights=filename_insights,
        health_score=health_score,
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
    od_analysis: ObjectDetectionAnalysis,
) -> ImageInsights:
    # Image size plot.
    plots.width_heigth_pixels_plot(
        output_file=output_folder / "image_size_plot.png",
        size_histogram=image_analysis.image_sizes,
        title="Image Sizes",
    )

    # Curated sample selection. We pick a mix of representative + edge cases
    # so a dataset reviewer sees what they need to act on.
    sample_folder = output_folder / "sample"
    sample_folder.mkdir(parents=True, exist_ok=True)
    sample_images = _select_sample_images(
        image_analysis=image_analysis, od_analysis=od_analysis
    )
    for sample in sample_images:
        src = image_analysis.image_folder / sample.filename
        dst = sample_folder / sample.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            shutil.copy2(src=src, dst=dst)

    return ImageInsights(
        image_sizes_most_common=list(image_analysis.image_sizes.most_common()),
        image_size_plot="image_size_plot.png",
        sample_images=sample_images,
    )


def _select_sample_images(
    image_analysis: ImageAnalysis,
    od_analysis: ObjectDetectionAnalysis,
    target_count: int = 8,
) -> List[SampleImage]:
    """Curated sample: mix of random + edge cases.

    Falls back gracefully to random-only when metadata is missing (e.g. an
    image had no label, so we don't know its object count).
    """
    filenames = sorted(image_analysis.filename_set)
    if not filenames:
        return []

    rng = random.Random(42)
    picks: List[Tuple[str, str]] = []  # (filename, label)
    seen: Set[str] = set()

    def add(filename: str, label: str) -> None:
        if filename and filename not in seen and filename in image_analysis.filename_set:
            seen.add(filename)
            picks.append((filename, label))

    # 1. Smallest and largest by pixel count; also extreme aspect ratios.
    if image_analysis.filename_to_size:
        by_pixels = sorted(
            image_analysis.filename_to_size.items(), key=lambda kv: kv[1][0] * kv[1][1]
        )
        add(by_pixels[0][0], "smallest")
        add(by_pixels[-1][0], "largest")
        widest = max(
            image_analysis.filename_to_size.items(),
            key=lambda kv: kv[1][0] / kv[1][1] if kv[1][1] else 0,
        )
        tallest = max(
            image_analysis.filename_to_size.items(),
            key=lambda kv: kv[1][1] / kv[1][0] if kv[1][0] else 0,
        )
        add(widest[0], "widest")
        add(tallest[0], "tallest")

    # 2. Most / zero objects (if label info is available).
    if od_analysis.objects_per_filename:
        by_count = sorted(
            od_analysis.objects_per_filename.items(), key=lambda kv: kv[1]
        )
        if by_count:
            add(by_count[-1][0], f"most objects ({by_count[-1][1]})")
            if by_count[0][1] == 0:
                add(by_count[0][0], "zero objects")

    # 3. Most-classes-present.
    if od_analysis.classes_per_filename:
        by_cls = max(
            od_analysis.classes_per_filename.items(), key=lambda kv: kv[1]
        )
        add(by_cls[0], f"most classes ({by_cls[1]})")

    # 4. Random fill.
    remaining = [f for f in filenames if f not in seen]
    rng.shuffle(remaining)
    for f in remaining:
        if len(picks) >= target_count:
            break
        add(f, "random")

    # If we still have fewer than target_count it's because the dataset is
    # smaller than target_count — that's fine, return what we have.
    return [
        SampleImage(filename=f, path=Path("./sample") / f, label=lbl)
        for f, lbl in picks[:target_count]
    ]


def _get_object_detection_insights(
    output_folder: Path,
    od_analysis: ObjectDetectionAnalysis,
    image_folder: Path,
    num_images: int,
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

    num_classes = len(od_analysis.classes)
    num_objects = od_analysis.total.num_objects
    avg_objects_per_image = (num_objects / num_images) if num_images else 0.0
    avg_objects_per_class = (num_objects / num_classes) if num_classes else 0.0
    avg_images_per_class = (num_images / num_classes) if num_classes else 0.0

    imbalance = _compute_imbalance_stats(
        [c.num_objects for c in od_analysis.classes.values()]
    )

    # Co-occurrence plot. Only renders when there's at least 2 classes and
    # at least one off-diagonal count (otherwise the chart is trivial).
    cooccurrence_plot_rel = ""
    if (
        od_analysis.cooccurrence_matrix is not None
        and od_analysis.cooccurrence_matrix.shape[0] > 1
    ):
        cooccurrence_path = output_folder / "cooccurrence.png"
        class_names = [
            od_analysis.classes[cid].class_name
            for cid in od_analysis.cooccurrence_class_ids
        ]
        plots.cooccurrence_plot(
            output_file=cooccurrence_path,
            matrix=od_analysis.cooccurrence_matrix,
            class_names=class_names,
        )
        cooccurrence_plot_rel = "cooccurrence.png"

    return ObjectDetectionInsights(
        num_classes=num_classes,
        class_ids_most_common=class_ids_most_common,
        plots=total_plots,
        class_plots=class_plots,
        avg_objects_per_image=avg_objects_per_image,
        avg_objects_per_class=avg_objects_per_class,
        avg_images_per_class=avg_images_per_class,
        imbalance=imbalance,
        cooccurrence_plot=cooccurrence_plot_rel,
    )


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_health_score(
    image_analysis: ImageAnalysis,
    od_analysis: ObjectDetectionAnalysis,
    leakage_groups: Optional[List[object]] = None,
) -> HealthScore:
    """Roll the dataset's metrics into a single 0-100 health read.

    `leakage_groups` is the output of `analyze.detect_cross_split_leakage`;
    when None we assume no splits were configured and skip the leakage
    subscore (weight redistributed proportionally).
    """
    subscores: List[Subscore] = []
    issues: List[Tuple[int, str]] = []  # (priority, text), lower = more urgent

    has_labels = od_analysis.total.num_objects > 0

    # --- Balance subscore ---
    if has_labels and len(od_analysis.classes) > 1:
        counts = [c.num_objects for c in od_analysis.classes.values()]
        imb = _compute_imbalance_stats(counts)
        balance_score = 100.0 * imb.normalized_entropy
        subscores.append(
            Subscore(
                name="Balance",
                score=balance_score,
                grade=_letter_grade(balance_score),
                detail=(
                    f"Normalized entropy {imb.normalized_entropy:.2f}, "
                    f"top class = {100 * imb.top_class_share:.0f} % of objects"
                ),
            )
        )
        if balance_score < 70:
            issues.append(
                (
                    10,
                    f"Class imbalance is high — top class holds "
                    f"{100 * imb.top_class_share:.0f} % of all objects. "
                    f"Consider resampling or class-weighted loss.",
                )
            )
    elif has_labels:
        subscores.append(
            Subscore(
                name="Balance",
                score=-1,
                grade="—",
                detail="Only 1 class — balance not applicable.",
            )
        )

    # --- Per-class data subscore ---
    # A detector's accuracy is bounded by its rarest class, so we score on
    # the minimum count across classes rather than the median. Any class
    # below the hard floor caps the subscore at D-grade regardless.
    if has_labels and od_analysis.classes:
        class_counts = [c.num_objects for c in od_analysis.classes.values()]
        min_count = min(class_counts)
        data_score = min(100.0, 100.0 * min_count / IDEAL_OBJECTS_PER_CLASS)
        starved = [
            c for c in od_analysis.classes.values() if c.num_objects < MIN_OBJECTS_PER_CLASS
        ]
        if starved:
            data_score = min(data_score, 40.0)
        detail = (
            f"rarest class has {min_count} objects "
            f"(ideal ≥ {IDEAL_OBJECTS_PER_CLASS})"
        )
        if starved:
            detail += f"; {len(starved)} class(es) below {MIN_OBJECTS_PER_CLASS}"
        subscores.append(
            Subscore(
                name="Per-class data",
                score=data_score,
                grade=_letter_grade(data_score),
                detail=detail,
            )
        )
        for cls in starved[:3]:
            issues.append(
                (
                    5,
                    f"Class “{cls.class_name}” has only {cls.num_objects} objects "
                    f"(< {MIN_OBJECTS_PER_CLASS}) — collect more samples or "
                    f"merge into a parent class.",
                )
            )

    # --- Annotation quality subscore ---
    if has_labels:
        total_objs = od_analysis.total.num_objects
        bad = (
            od_analysis.total.tiny_object_count
            + od_analysis.total.huge_object_count
            + len(od_analysis.duplicate_annotations)
            + len(od_analysis.class_conflicts)
        )
        ann_score = max(0.0, 100.0 * (1.0 - bad / total_objs)) if total_objs else 100.0
        subscores.append(
            Subscore(
                name="Annotation quality",
                score=ann_score,
                grade=_letter_grade(ann_score),
                detail=(
                    f"{bad} flagged / {total_objs} objects "
                    f"({100 * bad / total_objs:.1f} %)"
                    if total_objs
                    else "No objects."
                ),
            )
        )
        if od_analysis.class_conflicts:
            issues.append(
                (
                    15,
                    f"{len(od_analysis.class_conflicts)} class-conflict pair(s) "
                    f"(IoU ≥ 0.5, different classes) — likely mislabels.",
                )
            )
        if od_analysis.duplicate_annotations:
            issues.append(
                (
                    20,
                    f"{len(od_analysis.duplicate_annotations)} duplicate annotation "
                    f"pair(s) — remove the doubles.",
                )
            )
        if od_analysis.total.huge_object_count > 0:
            issues.append(
                (
                    25,
                    f"{od_analysis.total.huge_object_count} box(es) cover > 50 % "
                    f"of their image — verify they aren't whole-scene mislabels.",
                )
            )

    # --- Image quality subscore ---
    qf = image_analysis.quality_flags
    if qf is not None and qf.sample_size > 0:
        # One image can trip multiple flags; count distinct filenames.
        flagged_set = (
            set(qf.uniform_files)
            | set(qf.blurry_files)
            | set(qf.extreme_aspect_files)
            | set(image_analysis.corrupt_files)
        )
        flagged = len(flagged_set)
        # Corrupt files are NOT in the sample_size (they failed the size pass),
        # so we score against (sample_size + corrupt).
        denom = qf.sample_size + len(image_analysis.corrupt_files)
        img_score = max(0.0, 100.0 * (1.0 - flagged / denom)) if denom else 100.0
        subscores.append(
            Subscore(
                name="Image quality",
                score=img_score,
                grade=_letter_grade(img_score),
                detail=(
                    f"{flagged} flagged / {denom} images checked "
                    f"({100 * flagged / denom:.1f} %)"
                ),
            )
        )
        if image_analysis.corrupt_files:
            issues.append(
                (
                    1,
                    f"{len(image_analysis.corrupt_files)} image(s) are corrupt / "
                    f"unreadable — exclude or replace them.",
                )
            )
        if len(qf.blurry_files) > denom * 0.05:  # >5 % blurry
            issues.append(
                (
                    30,
                    f"{len(qf.blurry_files)} image(s) look blurry — review capture "
                    f"quality before training.",
                )
            )
    elif qf is not None:
        subscores.append(
            Subscore(
                name="Image quality",
                score=-1,
                grade="—",
                detail="No images inspected.",
            )
        )

    # --- Leakage subscore ---
    if leakage_groups is not None:
        leak_count = len(leakage_groups)
        leak_score = max(0.0, 100.0 - 20.0 * leak_count)
        subscores.append(
            Subscore(
                name="Cross-split leakage",
                score=leak_score,
                grade=_letter_grade(leak_score),
                detail=(
                    "No near-duplicates across splits."
                    if leak_count == 0
                    else f"{leak_count} near-duplicate group(s) span multiple splits."
                ),
            )
        )
        if leak_count > 0:
            issues.append(
                (
                    0,  # highest priority — invalidates eval
                    f"{leak_count} near-duplicate group(s) found across splits — "
                    f"this inflates eval accuracy. Dedupe before reporting numbers.",
                )
            )

    # --- Overall ---
    computable = [s for s in subscores if s.score >= 0]
    if not computable:
        overall: Optional[float] = None
        overall_grade = "—"
    else:
        # Reweight only over computable subscores.
        weights = {
            "Balance": HEALTH_WEIGHTS["balance"],
            "Per-class data": HEALTH_WEIGHTS["per_class_data"],
            "Annotation quality": HEALTH_WEIGHTS["annotation_quality"],
            "Image quality": HEALTH_WEIGHTS["image_quality"],
            "Cross-split leakage": HEALTH_WEIGHTS["leakage"],
        }
        total_weight = sum(weights[s.name] for s in computable)
        overall = sum(weights[s.name] * s.score for s in computable) / total_weight
        overall_grade = _letter_grade(overall)

    issues.sort(key=lambda p: p[0])
    ordered_issues = [msg for _, msg in issues[:5]]

    return HealthScore(
        overall=overall,
        grade=overall_grade,
        subscores=subscores,
        issues=ordered_issues,
    )


def _compute_imbalance_stats(class_counts: List[int]) -> ImbalanceStats:
    """Entropy, Gini, top-class share, under-represented count for OD classes.

    Classes with zero objects are excluded from entropy/Gini (they don't
    represent a real slice of the dataset) but still counted for the
    under-represented count.
    """
    nonzero = [c for c in class_counts if c > 0]
    total = sum(nonzero)
    if total == 0 or not nonzero:
        return ImbalanceStats(
            entropy=0.0,
            normalized_entropy=0.0,
            gini=0.0,
            top_class_share=0.0,
            under_represented_count=0,
        )

    probs = [c / total for c in nonzero]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(nonzero)) if len(nonzero) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    sorted_counts = sorted(nonzero)
    n = len(sorted_counts)
    cum = sum((i + 1) * c for i, c in enumerate(sorted_counts))
    gini = (2 * cum) / (n * total) - (n + 1) / n if n > 1 else 0.0

    top_class_share = max(nonzero) / total
    under_represented_count = sum(
        1 for c in class_counts if c / total < UNDER_REPRESENTED_FRACTION
    )

    return ImbalanceStats(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        gini=gini,
        top_class_share=top_class_share,
        under_represented_count=under_represented_count,
    )


def _get_filename_insights(
    output_folder: Path,
    image_filename_set: Set[str],
    label_filename_set: Set[str],
) -> FilenameInsights:
    # Match by stem so that different extensions and subdirectory prefixes
    # produced by Labelformat don't create spurious no-label / no-image
    # entries.
    def stem(name: str) -> str:
        return Path(name).stem

    image_stems = {stem(f): f for f in image_filename_set}
    label_stems = {stem(f): f for f in label_filename_set}

    if len(image_stems) < len(image_filename_set) or len(label_stems) < len(
        label_filename_set
    ):
        logger.warning(
            "Filename stems collide after normalization; reported no-label / "
            "no-image counts may be inaccurate. Check for duplicate basenames "
            "across subdirectories."
        )

    missing_label_stems = sorted(image_stems.keys() - label_stems.keys())
    missing_image_stems = sorted(label_stems.keys() - image_stems.keys())
    filenames_no_label = [image_stems[s] for s in missing_label_stems]
    filenames_no_image = [label_stems[s] for s in missing_image_stems]

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
