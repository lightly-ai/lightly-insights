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

# Image-quality thresholds.
UNIFORM_LUMINANCE_STD = 2.0  # lower than this -> all-black/all-white
BLUR_LAPLACIAN_VAR = 50.0  # lower than this -> likely blurry
EXTREME_ASPECT_RATIO = 5.0  # w/h > 5 or < 0.2 -> aspect outlier
QUALITY_SAMPLE_CAP = 1000  # images we scan for luminance/blur

# Class-inconsistent box detection (different classes, overlapping boxes).
CLASS_CONFLICT_IOU = 0.5  # IoU >= this with different classes -> possible mislabel

# Anchor recommendation.
ANCHOR_DEFAULT_K = 9  # YOLO-style 9-anchor default


@dataclass(frozen=True)
class QualityFlags:
    """Per-image quality red flags, sampled on large datasets."""

    sample_size: int  # how many images were inspected
    uniform_files: List[str] = field(default_factory=list)  # all-black/all-white
    blurry_files: List[str] = field(default_factory=list)
    extreme_aspect_files: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageAnalysis:
    num_images: int
    image_folder: Path
    filename_set: Set[str]
    image_sizes: Counter[Tuple[int, int]]
    median_size: Tuple[int, int]
    corrupt_files: List[str] = field(default_factory=list)
    filename_to_size: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    quality_flags: Optional[QualityFlags] = None
    # Groups of files that hash to very-similar values (near-duplicates).
    # Empty when imagehash is not installed.
    near_duplicate_groups: List[List[str]] = field(default_factory=list)


@dataclass(frozen=True)
class DuplicatePair:
    filename: str
    box_a: Tuple[float, float, float, float]
    box_b: Tuple[float, float, float, float]
    iou: float
    category_a: str
    category_b: str


@dataclass(frozen=True)
class ClassConflictPair:
    """Same-image boxes from *different* classes with high IoU.

    Typically indicates a labeling mistake — the same object annotated under
    two different classes.
    """

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
    # Counts of images where classes[i] and classes[j] both appear. The
    # `cooccurrence_class_ids` list defines the row/column order.
    cooccurrence_matrix: Optional[NDArray[np.int_]] = None
    cooccurrence_class_ids: List[int] = field(default_factory=list)
    # Per-image filename -> number of objects. Used downstream for curated
    # sample selection.
    objects_per_filename: Dict[str, int] = field(default_factory=dict)
    classes_per_filename: Dict[str, int] = field(default_factory=dict)
    # Boxes from different classes with high IoU — probable mislabels.
    class_conflicts: List[ClassConflictPair] = field(default_factory=list)
    # Recommended anchor sizes in pixels, sorted by area (smallest first).
    recommended_anchors: List[Tuple[float, float]] = field(default_factory=list)


def _read_image_size(
    image_path: Path,
) -> Tuple[Path, Optional[Tuple[int, int]], Optional[str]]:
    """Open an image and return its size. On failure, return an error message."""
    try:
        with Image.open(image_path) as image:
            return image_path, image.size, None
    except (UnidentifiedImageError, OSError) as exc:
        return image_path, None, str(exc)


def _laplacian_variance(gray: NDArray[np.float_]) -> float:
    """Variance of the Laplacian. Standard blur-detection proxy."""
    # 3x3 Laplacian kernel applied via numpy. Avoids scipy dependency.
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    out = np.zeros_like(gray)
    # Manual 2D convolution at interior pixels (edges stay 0 — good enough for variance).
    out[1:-1, 1:-1] = (
        k[1, 1] * gray[1:-1, 1:-1]
        + k[0, 1] * gray[:-2, 1:-1]
        + k[2, 1] * gray[2:, 1:-1]
        + k[1, 0] * gray[1:-1, :-2]
        + k[1, 2] * gray[1:-1, 2:]
    )
    return float(out.var())


def _inspect_quality(
    image_path: Path, rel_name: str
) -> Tuple[str, bool, bool, bool]:
    """Return quality signals for a single image.

    Returns (rel_name, is_uniform, is_blurry, is_extreme_aspect).
    """
    try:
        with Image.open(image_path) as img:
            # Downsample first — blur metric + std are stable at small sizes
            # and this keeps big images fast.
            w, h = img.size
            thumb = img.convert("L")
            thumb.thumbnail((256, 256))
            arr = np.asarray(thumb, dtype=np.float64)
    except (UnidentifiedImageError, OSError):
        return rel_name, False, False, False
    is_uniform = bool(arr.std() < UNIFORM_LUMINANCE_STD) if arr.size else False
    is_blurry = bool(_laplacian_variance(arr) < BLUR_LAPLACIAN_VAR)
    ratio = (w / h) if h > 0 else 0
    is_extreme_aspect = ratio > EXTREME_ASPECT_RATIO or (
        ratio > 0 and ratio < 1 / EXTREME_ASPECT_RATIO
    )
    return rel_name, is_uniform, is_blurry, is_extreme_aspect


def _compute_quality_flags(
    paths_and_names: List[Tuple[Path, str]],
    max_workers: int,
    sample_cap: int = QUALITY_SAMPLE_CAP,
) -> QualityFlags:
    """Sample up to `sample_cap` images and check for red flags.

    We sample rather than scan every image because the checks open pixels
    (not just headers) and we want to stay snappy on 100k-image datasets.
    """
    import random as _random

    if not paths_and_names:
        return QualityFlags(sample_size=0)
    if len(paths_and_names) > sample_cap:
        rng = _random.Random(42)
        sample = rng.sample(paths_and_names, sample_cap)
    else:
        sample = list(paths_and_names)

    uniform: List[str] = []
    blurry: List[str] = []
    extreme: List[str] = []

    def _task(pn: Tuple[Path, str]) -> Tuple[str, bool, bool, bool]:
        return _inspect_quality(pn[0], pn[1])

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rel_name, is_uniform, is_blurry, is_ext in tqdm.tqdm(
            pool.map(_task, sample),
            total=len(sample),
            desc="Scanning quality",
            unit="images",
        ):
            if is_uniform:
                uniform.append(rel_name)
            if is_blurry:
                blurry.append(rel_name)
            if is_ext:
                extreme.append(rel_name)

    return QualityFlags(
        sample_size=len(sample),
        uniform_files=sorted(uniform),
        blurry_files=sorted(blurry),
        extreme_aspect_files=sorted(extreme),
    )


def _compute_near_duplicates(
    paths_and_names: List[Tuple[Path, str]],
    max_workers: int,
    hamming_threshold: int = 5,
) -> List[List[str]]:
    """Group near-duplicate images via dHash. Returns [] if imagehash missing.

    Optional dependency so the core insights tool stays lean. We group
    filenames whose dHash Hamming distance is <= hamming_threshold.
    """
    try:
        import imagehash  # type: ignore[import]
    except ImportError:
        logger.info(
            "imagehash not installed; skipping near-duplicate image detection."
        )
        return []

    def _hash(pn: Tuple[Path, str]) -> Tuple[str, Optional[object]]:
        try:
            with Image.open(pn[0]) as img:
                return pn[1], imagehash.dhash(img)
        except (UnidentifiedImageError, OSError):
            return pn[1], None

    hashes: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rel_name, h in tqdm.tqdm(
            pool.map(_hash, paths_and_names),
            total=len(paths_and_names),
            desc="Hashing for near-duplicates",
            unit="images",
        ):
            if h is not None:
                hashes[rel_name] = h

    # Group via union-find on pairwise Hamming distance. O(n^2) — fine for
    # a few thousand images, would need LSH for millions.
    names = sorted(hashes.keys())
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if hashes[a] - hashes[b] <= hamming_threshold:  # type: ignore[operator]
                union(a, b)

    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    # Only return groups with >1 member.
    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=lambda g: -len(g),
    )


def analyze_images(
    image_folder: Path,
    max_workers: int = 16,
    recursive: bool = False,
    check_quality: bool = True,
    find_near_duplicates: bool = False,
) -> ImageAnalysis:
    filename_set: Set[str] = set()
    image_sizes = Counter[Tuple[int, int]]()
    image_widths: List[int] = []
    image_heights: List[int] = []
    corrupt_files: List[str] = []
    filename_to_size: Dict[str, Tuple[int, int]] = {}
    good_paths: List[Tuple[Path, str]] = []  # for downstream quality checks

    logger.info(
        f"Listing images in {image_folder} "
        f"({'recursively' if recursive else 'top-level only'})."
    )
    glob_iter = image_folder.rglob("*.*") if recursive else image_folder.glob("*.*")
    sorted_paths = sorted(
        path for path in glob_iter if path.suffix.lower() in IMAGE_EXTENSIONS
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
            # In recursive mode, use relative path so sub/foo.jpg and
            # sub2/foo.jpg don't collide; otherwise keep the old basename-only
            # behavior to stay backwards-compatible.
            rel_name = (
                str(image_path.relative_to(image_folder))
                if recursive
                else image_path.name
            )
            if error is not None or size is None:
                corrupt_files.append(rel_name)
                logger.warning(
                    f"Could not read {rel_name}: {error or 'unknown error'}"
                )
                continue
            filename_set.add(rel_name)
            filename_to_size[rel_name] = size
            image_sizes[size] += 1
            image_widths.append(size[0])
            image_heights.append(size[1])
            good_paths.append((image_path, rel_name))

    # Note: width and height medians are computed independently, so the pair
    # is not guaranteed to correspond to any single image in the dataset.
    num_images = len(filename_set)
    median_size = (
        int(np.median(image_widths)) if num_images > 0 else 0,
        int(np.median(image_heights)) if num_images > 0 else 0,
    )

    quality_flags = (
        _compute_quality_flags(good_paths, max_workers=max_workers)
        if check_quality
        else None
    )
    near_duplicate_groups = (
        _compute_near_duplicates(good_paths, max_workers=max_workers)
        if find_near_duplicates
        else []
    )

    return ImageAnalysis(
        num_images=num_images,
        image_folder=image_folder,
        filename_set=filename_set,
        image_sizes=image_sizes,
        median_size=median_size,
        corrupt_files=sorted(corrupt_files),
        filename_to_size=filename_to_size,
        quality_flags=quality_flags,
        near_duplicate_groups=near_duplicate_groups,
    )


@dataclass(frozen=True)
class LeakageGroup:
    """A cluster of near-identical images spread across multiple splits."""

    # Mapping split name -> filenames in that split that are near-duplicates
    # of the other entries in this group.
    files_by_split: Dict[str, List[str]]
    # Number of distinct splits this cluster touches (>= 2 for it to be a leak).
    num_splits: int


def detect_cross_split_leakage(
    analyses_by_split: Dict[str, "ImageAnalysis"],
    hamming_threshold: int = 5,
    max_workers: int = 16,
) -> List[LeakageGroup]:
    """Find images present in more than one split via perceptual hashing.

    Takes the result of running `analyze_images` on each split folder. Hashes
    every image, unions near-duplicates, and returns groups that span at
    least two splits. Empty when `imagehash` is not installed.

    Typical usage:

        analyses = {
            split: analyze.analyze_images(folder)
            for split, folder in {"train": ..., "val": ..., "test": ...}.items()
        }
        leaks = analyze.detect_cross_split_leakage(analyses)
    """
    try:
        import imagehash  # type: ignore[import]
    except ImportError:
        logger.warning(
            "imagehash not installed; cannot detect cross-split leakage. "
            "Install the 'near-duplicates' extra."
        )
        return []

    # Collect (split, filename, full_path) tuples.
    entries: List[Tuple[str, str, Path]] = []
    for split_name, analysis in analyses_by_split.items():
        for rel_name in sorted(analysis.filename_set):
            entries.append(
                (split_name, rel_name, analysis.image_folder / rel_name)
            )

    def _hash(entry: Tuple[str, str, Path]) -> Tuple[str, str, Optional[object]]:
        split, name, path = entry
        try:
            with Image.open(path) as img:
                return split, name, imagehash.dhash(img)
        except (UnidentifiedImageError, OSError):
            return split, name, None

    hashed: List[Tuple[str, str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for split, name, h in tqdm.tqdm(
            pool.map(_hash, entries),
            total=len(entries),
            desc="Hashing splits for leakage detection",
            unit="images",
        ):
            if h is not None:
                hashed.append((split, name, h))

    # Union-find across all (split, filename) pairs.
    keys = [(s, n) for s, n, _ in hashed]
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {k: k for k in keys}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[str, str], b: Tuple[str, str]) -> None:
        parent[find(a)] = find(b)

    for i in range(len(hashed)):
        s_i, n_i, h_i = hashed[i]
        for j in range(i + 1, len(hashed)):
            s_j, n_j, h_j = hashed[j]
            if h_i - h_j <= hamming_threshold:  # type: ignore[operator]
                union((s_i, n_i), (s_j, n_j))

    # Bucket members by root. Only return groups touching 2+ splits.
    groups: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    result: List[LeakageGroup] = []
    for members in groups.values():
        by_split: Dict[str, List[str]] = {}
        for split, name in members:
            by_split.setdefault(split, []).append(name)
        if len(by_split) >= 2:
            # Sort filenames inside each split for stable output.
            for split in by_split:
                by_split[split].sort()
            result.append(
                LeakageGroup(files_by_split=by_split, num_splits=len(by_split))
            )
    # Largest leaks first.
    result.sort(key=lambda g: -sum(len(v) for v in g.files_by_split.values()))
    return result


def _kmeans_anchors(
    sizes: List[Tuple[float, float]],
    k: int = ANCHOR_DEFAULT_K,
    iters: int = 30,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Compute k anchor sizes via k-means clustering on (w, h) pairs.

    Pure numpy, no sklearn dependency. Returns anchors sorted by area
    (smallest first). Empty list if `sizes` has fewer than `k` entries.
    """
    if len(sizes) < k:
        return []
    rng = np.random.default_rng(seed)
    data = np.asarray(sizes, dtype=np.float64)
    # k-means++ init: pick one point, then each next point with probability
    # proportional to squared distance from its nearest existing center.
    centers = np.empty((k, 2), dtype=np.float64)
    centers[0] = data[rng.integers(0, len(data))]
    for i in range(1, k):
        d2 = np.min(
            np.sum((data[:, None, :] - centers[:i]) ** 2, axis=2), axis=1
        )
        total = d2.sum()
        if total == 0:
            centers[i] = data[rng.integers(0, len(data))]
            continue
        probs = d2 / total
        idx = int(rng.choice(len(data), p=probs))
        centers[i] = data[idx]

    for _ in range(iters):
        # Assign each point to nearest center.
        dists = np.sum((data[:, None, :] - centers) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        # Recompute centers (median is more robust than mean for anchors).
        new_centers = centers.copy()
        for c in range(k):
            pts = data[labels == c]
            if len(pts) > 0:
                new_centers[c] = np.median(pts, axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    # Sort by area.
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    return [(float(centers[i, 0]), float(centers[i, 1])) for i in order]


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
    class_conflicts: List[ClassConflictPair] = []
    total_data = ClassAnalysis.create_empty(id=-1, name="[All classes]")
    class_data = {
        category.id: ClassAnalysis.create_empty(id=category.id, name=category.name)
        for category in label_input.get_categories()
    }
    # Cache category id list so we don't re-materialize per label.
    category_ids = [cat.id for cat in label_input.get_categories()]
    category_id_to_index = {cid: idx for idx, cid in enumerate(category_ids)}
    num_classes = len(category_ids)
    cooccurrence = (
        np.zeros((num_classes, num_classes), dtype=np.int_) if num_classes > 0 else None
    )
    objects_per_filename: Dict[str, int] = {}
    classes_per_filename: Dict[str, int] = {}

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

        # Pairwise IoU within this image: catches duplicates (same class,
        # near-identical box) and class conflicts (different class, high IoU).
        for i in range(len(label.objects)):
            for j in range(i + 1, len(label.objects)):
                obj_a = label.objects[i]
                obj_b = label.objects[j]
                iou = _box_iou(obj_a.box, obj_b.box)
                if iou < CLASS_CONFLICT_IOU:
                    continue
                box_a = obj_a.box
                box_b = obj_b.box
                box_a_tuple = (box_a.xmin, box_a.ymin, box_a.xmax, box_a.ymax)
                box_b_tuple = (box_b.xmin, box_b.ymin, box_b.xmax, box_b.ymax)
                if obj_a.category.id == obj_b.category.id:
                    if iou >= DUPLICATE_IOU_THRESHOLD:
                        duplicate_annotations.append(
                            DuplicatePair(
                                filename=label.image.filename,
                                box_a=box_a_tuple,
                                box_b=box_b_tuple,
                                iou=iou,
                                category_a=obj_a.category.name,
                                category_b=obj_b.category.name,
                            )
                        )
                else:
                    class_conflicts.append(
                        ClassConflictPair(
                            filename=label.image.filename,
                            box_a=box_a_tuple,
                            box_b=box_b_tuple,
                            iou=iou,
                            category_a=obj_a.category.name,
                            category_b=obj_b.category.name,
                        )
                    )

        # Class co-occurrence: which classes appear together in this image.
        present_ids = {
            obj.category.id for obj in label.objects if obj.category.id in category_id_to_index
        }
        if cooccurrence is not None:
            present_indices = sorted(category_id_to_index[cid] for cid in present_ids)
            for a_idx in present_indices:
                for b_idx in present_indices:
                    cooccurrence[a_idx, b_idx] += 1

        # Per-image counters used by present.py for curated sample selection.
        objects_per_filename[label.image.filename] = len(label.objects)
        classes_per_filename[label.image.filename] = len(present_ids)

        # Update objects per image for classes.
        for category_id in category_ids:
            class_data[category_id].objects_per_image[
                num_objects_per_category[category_id]
            ] += 1

    recommended_anchors = _kmeans_anchors(total_data.object_sizes_abs)

    return ObjectDetectionAnalysis(
        num_images=num_images,
        num_images_zero_objects=num_images_zero_objects,
        filename_set=filename_set,
        total=total_data,
        classes=class_data,
        duplicate_annotations=duplicate_annotations,
        cooccurrence_matrix=cooccurrence,
        cooccurrence_class_ids=list(category_ids),
        objects_per_filename=objects_per_filename,
        classes_per_filename=classes_per_filename,
        class_conflicts=class_conflicts,
        recommended_anchors=recommended_anchors,
    )
