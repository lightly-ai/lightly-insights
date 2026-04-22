import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Counter, Dict, List, Optional, Set, Tuple

import numpy as np
import tqdm
from labelformat.model.bounding_box import BoundingBox
from labelformat.model.object_detection import ObjectDetectionInput
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

HEATMAP_SIZE = 100

# Quality thresholds.
TINY_OBJECT_REL_AREA = 0.005  # <0.5% of image area -> "tiny"
HUGE_OBJECT_REL_AREA = 0.5  # >50% of image area -> "huge"
EDGE_TOUCH_PIXELS = 1.0  # boxes within this many pixels of any image edge
DUPLICATE_IOU_THRESHOLD = 0.9  # boxes above this IoU are flagged as duplicates


@dataclass(frozen=True)
class ImageAnalysis:
    num_images: int
    image_folder: Path
    filename_set: Set[str]
    image_sizes: Counter[Tuple[int, int]]
    median_size: Tuple[int, int]
    corrupt_files: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DuplicatePair:
    filename: str
    box_a: Tuple[float, float, float, float]
    box_b: Tuple[float, float, float, float]
    iou: float
    category_a: str
    category_b: str


@dataclass
class ClassAnalysis:
    class_id: int
    class_name: str

    num_objects: int
    objects_per_image: Counter[int]
    object_sizes_abs: List[Tuple[float, float]]
    object_sizes_rel: List[Tuple[float, float]]
    heatmap: NDArray[np.float_]

    sample_filenames: List[str]

    # Quality signals. Populated during the per-object loop.
    aspect_ratios: List[float] = field(default_factory=list)
    tiny_object_count: int = 0
    huge_object_count: int = 0
    edge_touching_count: int = 0

    # Derived stats.

    @property
    def avg_size(self) -> Tuple[float, float]:
        if self.num_objects == 0:
            return (0.0, 0.0)
        else:
            return (
                sum(w for w, _ in self.object_sizes_abs) / self.num_objects,
                sum(h for _, h in self.object_sizes_abs) / self.num_objects,
            )

    @property
    def avg_rel_area(self) -> float:
        if self.num_objects == 0:
            return 0.0
        else:
            return sum(w * h for w, h in self.object_sizes_rel) / self.num_objects

    @classmethod
    def create_empty(cls, id: int, name: str) -> "ClassAnalysis":
        return cls(
            class_id=id,
            class_name=name,
            num_objects=0,
            objects_per_image=Counter(),
            object_sizes_abs=[],
            object_sizes_rel=[],
            heatmap=np.zeros((HEATMAP_SIZE, HEATMAP_SIZE)),
            sample_filenames=[],
        )


@dataclass(frozen=True)
class ObjectDetectionAnalysis:
    num_images: int
    num_images_zero_objects: int
    filename_set: Set[str]
    total: ClassAnalysis
    classes: Dict[int, ClassAnalysis]
    duplicate_annotations: List[DuplicatePair] = field(default_factory=list)


def _read_image_size(
    image_path: Path,
) -> Tuple[Path, Optional[Tuple[int, int]], Optional[str]]:
    """Open an image and return its size. On failure, return an error message."""
    try:
        with Image.open(image_path) as image:
            return image_path, image.size, None
    except (UnidentifiedImageError, OSError) as exc:
        return image_path, None, str(exc)


def analyze_images(image_folder: Path, max_workers: int = 16) -> ImageAnalysis:
    filename_set: Set[str] = set()
    image_sizes = Counter[Tuple[int, int]]()
    image_widths: List[int] = []
    image_heights: List[int] = []
    corrupt_files: List[str] = []

    # Currently we list non-recursively. We could add a flag to allow
    # recursive listing in the future.
    logger.info(f"Listing images in {image_folder}.")
    sorted_paths = sorted(
        path
        for path in image_folder.glob("*.*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"Found {len(sorted_paths)} images.")

    # PIL releases the GIL during image I/O, so threads give a real speedup on
    # disk-bound workloads. Keeps memory usage bounded since we only hold one
    # size tuple per image at a time.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for image_path, size, error in tqdm.tqdm(
            pool.map(_read_image_size, sorted_paths),
            total=len(sorted_paths),
            desc="Reading image sizes",
            unit="images",
        ):
            if error is not None or size is None:
                corrupt_files.append(image_path.name)
                logger.warning(
                    f"Could not read {image_path.name}: {error or 'unknown error'}"
                )
                continue
            filename_set.add(image_path.name)
            image_sizes[size] += 1
            image_widths.append(size[0])
            image_heights.append(size[1])

    # Note: width and height medians are computed independently, so the pair
    # is not guaranteed to correspond to any single image in the dataset.
    num_images = len(filename_set)
    median_size = (
        int(np.median(image_widths)) if num_images > 0 else 0,
        int(np.median(image_heights)) if num_images > 0 else 0,
    )

    return ImageAnalysis(
        num_images=num_images,
        image_folder=image_folder,
        filename_set=filename_set,
        image_sizes=image_sizes,
        median_size=median_size,
        corrupt_files=sorted(corrupt_files),
    )


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two bounding boxes in the same image."""
    x1 = max(a.xmin, b.xmin)
    y1 = max(a.ymin, b.ymin)
    x2 = min(a.xmax, b.xmax)
    y2 = min(a.ymax, b.ymax)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin)
    area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def analyze_object_detections(
    label_input: ObjectDetectionInput,
) -> ObjectDetectionAnalysis:
    num_images = 0
    num_images_zero_objects = 0
    filename_set = set()
    duplicate_annotations: List[DuplicatePair] = []
    total_data = ClassAnalysis.create_empty(id=-1, name="[All classes]")
    class_data = {
        category.id: ClassAnalysis.create_empty(id=category.id, name=category.name)
        for category in label_input.get_categories()
    }
    # Cache category id list so we don't re-materialize per label.
    category_ids = [cat.id for cat in label_input.get_categories()]

    # Iterate over labels and count objects.
    for label in tqdm.tqdm(
        label_input.get_labels(),
        desc="Reading object detection labels",
        unit="labels",
    ):
        num_images += 1
        if len(label.objects) == 0:
            num_images_zero_objects += 1
        filename_set.add(label.image.filename)

        total_data.num_objects += len(label.objects)
        total_data.objects_per_image[len(label.objects)] += 1

        num_objects_per_category = Counter[int]()

        for obj in label.objects:
            class_datum = class_data[obj.category.id]

            # Number of objects.
            class_datum.num_objects += 1
            num_objects_per_category[obj.category.id] += 1

            # Object sizes.
            obj_w = obj.box.xmax - obj.box.xmin
            obj_h = obj.box.ymax - obj.box.ymin
            obj_size_abs = (obj_w, obj_h)
            obj_size_rel = (
                obj_w / label.image.width,
                obj_h / label.image.height,
            )
            total_data.object_sizes_abs.append(obj_size_abs)
            total_data.object_sizes_rel.append(obj_size_rel)
            class_datum.object_sizes_abs.append(obj_size_abs)
            class_datum.object_sizes_rel.append(obj_size_rel)

            # Aspect ratio. Guard against zero-height boxes.
            if obj_h > 0:
                ratio = obj_w / obj_h
                class_datum.aspect_ratios.append(ratio)
                total_data.aspect_ratios.append(ratio)

            # Tiny / huge flags based on relative area.
            rel_area = obj_size_rel[0] * obj_size_rel[1]
            if rel_area < TINY_OBJECT_REL_AREA:
                class_datum.tiny_object_count += 1
                total_data.tiny_object_count += 1
            if rel_area > HUGE_OBJECT_REL_AREA:
                class_datum.huge_object_count += 1
                total_data.huge_object_count += 1

            # Edge-touching: box sits within EDGE_TOUCH_PIXELS of any edge.
            touches_edge = (
                obj.box.xmin <= EDGE_TOUCH_PIXELS
                or obj.box.ymin <= EDGE_TOUCH_PIXELS
                or obj.box.xmax >= label.image.width - EDGE_TOUCH_PIXELS
                or obj.box.ymax >= label.image.height - EDGE_TOUCH_PIXELS
            )
            if touches_edge:
                class_datum.edge_touching_count += 1
                total_data.edge_touching_count += 1

            # Heatmap. Use floor/ceil + clamp so sub-cell boxes still
            # contribute to at least one cell.
            x1 = max(
                0,
                floor(obj.box.xmin / label.image.width * HEATMAP_SIZE),
            )
            x2 = min(
                HEATMAP_SIZE,
                ceil(obj.box.xmax / label.image.width * HEATMAP_SIZE),
            )
            y1 = max(
                0,
                floor(obj.box.ymin / label.image.height * HEATMAP_SIZE),
            )
            y2 = min(
                HEATMAP_SIZE,
                ceil(obj.box.ymax / label.image.height * HEATMAP_SIZE),
            )
            if x2 <= x1:
                x2 = min(HEATMAP_SIZE, x1 + 1)
            if y2 <= y1:
                y2 = min(HEATMAP_SIZE, y1 + 1)
            total_data.heatmap[y1:y2, x1:x2] += 1
            class_datum.heatmap[y1:y2, x1:x2] += 1

            # Sample images.
            if (
                len(class_datum.sample_filenames) < 4
                and label.image.filename not in class_datum.sample_filenames
            ):
                class_datum.sample_filenames.append(label.image.filename)

        # Duplicate-annotation detection (pairwise IoU within this image).
        for i in range(len(label.objects)):
            for j in range(i + 1, len(label.objects)):
                iou = _box_iou(label.objects[i].box, label.objects[j].box)
                if iou >= DUPLICATE_IOU_THRESHOLD:
                    box_a = label.objects[i].box
                    box_b = label.objects[j].box
                    duplicate_annotations.append(
                        DuplicatePair(
                            filename=label.image.filename,
                            box_a=(box_a.xmin, box_a.ymin, box_a.xmax, box_a.ymax),
                            box_b=(box_b.xmin, box_b.ymin, box_b.xmax, box_b.ymax),
                            iou=iou,
                            category_a=label.objects[i].category.name,
                            category_b=label.objects[j].category.name,
                        )
                    )

        # Update objects per image for classes.
        for category_id in category_ids:
            class_data[category_id].objects_per_image[
                num_objects_per_category[category_id]
            ] += 1

    return ObjectDetectionAnalysis(
        num_images=num_images,
        num_images_zero_objects=num_images_zero_objects,
        filename_set=filename_set,
        total=total_data,
        classes=class_data,
        duplicate_annotations=duplicate_annotations,
    )
