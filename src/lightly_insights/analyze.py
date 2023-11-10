import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Dict, List, Set, Tuple

import numpy as np
import tqdm
from labelformat.model.object_detection import ObjectDetectionInput
from numpy.typing import NDArray
from PIL import Image

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


@dataclass(frozen=True)
class ImageAnalysis:
    num_images: int
    image_folder: Path
    filename_set: Set[str]
    image_sizes: Counter[Tuple[int, int]]
    median_size: Tuple[int, int]


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


def analyze_images(image_folder: Path) -> ImageAnalysis:
    num_images = 0
    filename_set = set()

    image_sizes = Counter[Tuple[int, int]]()
    image_widths = []
    image_heights = []

    # Currently we list non-recursively. We could add a flag to allow
    # recursive listing in the future.
    logger.info(f"Listing images in {image_folder}.")
    sorted_paths = sorted(
        path
        for path in image_folder.glob("*.*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"Found {len(sorted_paths)} images.")

    for image_path in sorted_paths:
        num_images += 1
        filename_set.add(image_path.name)
        with Image.open(image_path) as image:
            image_sizes[image.size] += 1
            image_widths.append(image.size[0])
            image_heights.append(image.size[1])

    median_size = (
        sorted(image_widths)[num_images // 2] if num_images > 0 else 0,
        sorted(image_heights)[num_images // 2] if num_images > 0 else 0,
    )

    return ImageAnalysis(
        num_images=num_images,
        image_folder=image_folder,
        filename_set=filename_set,
        image_sizes=image_sizes,
        median_size=median_size,
    )


def analyze_object_detections(
    label_input: ObjectDetectionInput,
) -> ObjectDetectionAnalysis:
    num_images = 0
    num_images_zero_objects = 0
    filename_set = set()
    total_data = ClassAnalysis.create_empty(id=-1, name="[All classes]")
    class_data = {
        category.id: ClassAnalysis.create_empty(id=category.id, name=category.name)
        for category in label_input.get_categories()
    }

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
            obj_size_abs = (
                obj.box.xmax - obj.box.xmin,
                obj.box.ymax - obj.box.ymin,
            )
            obj_size_rel = (
                (obj.box.xmax - obj.box.xmin) / label.image.width,
                (obj.box.ymax - obj.box.ymin) / label.image.height,
            )
            total_data.object_sizes_abs.append(obj_size_abs)
            total_data.object_sizes_rel.append(obj_size_rel)
            class_datum.object_sizes_abs.append(obj_size_abs)
            class_datum.object_sizes_rel.append(obj_size_rel)

            # Heatmap.
            x1 = obj.box.xmin / label.image.width * HEATMAP_SIZE
            x2 = obj.box.xmax / label.image.width * HEATMAP_SIZE
            y1 = obj.box.ymin / label.image.height * HEATMAP_SIZE
            y2 = obj.box.ymax / label.image.height * HEATMAP_SIZE
            total_data.heatmap[int(y1) : int(y2), int(x1) : int(x2)] += 1
            class_datum.heatmap[int(y1) : int(y2), int(x1) : int(x2)] += 1

            # Sample images.
            if (
                len(class_datum.sample_filenames) < 4
                and label.image.filename not in class_datum.sample_filenames
            ):
                class_datum.sample_filenames.append(label.image.filename)

        # Update objects per image for classes.
        for category in label_input.get_categories():
            class_data[category.id].objects_per_image[
                num_objects_per_category[category.id]
            ] += num_objects_per_category[category.id]

    return ObjectDetectionAnalysis(
        num_images=num_images,
        num_images_zero_objects=num_images_zero_objects,
        filename_set=filename_set,
        total=total_data,
        classes=class_data,
    )
