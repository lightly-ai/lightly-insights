"""Dataset / Annotation domain model.

A ``Dataset`` is the canonical thing every ``Check`` consumes. Construction
is explicit (callers pass in-memory lists); adapters in
``lightly_insights.core.adapter`` populate it from the existing
``ImageAnalysis`` / ``ObjectDetectionAnalysis`` results so the refactor can
land incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from lightly_insights.core.geometry import Box, Mask, Polygon


class AnnotationKind(str, Enum):
    BOX = "box"
    POLYGON = "polygon"
    MASK = "mask"


# Shape of `Annotation.geometry`. Kept as a module-level alias so callers
# can pattern-match without reaching into geometry.
Geometry = Union[Box, Polygon, Mask]


@dataclass(frozen=True)
class Category:
    id: int
    name: str


@dataclass(frozen=True)
class Image:
    filename: str
    width: int
    height: int
    # Absolute path on disk, if available. Optional so datasets can be
    # constructed from metadata alone (e.g. a remote dataset manifest).
    path: Optional[Path] = None

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Annotation:
    """A single labeled object on an image.

    ``annotation_id`` is a stable integer assigned by the Dataset
    constructor — Findings reference annotations by id so reporters can
    look them up.
    """

    annotation_id: int
    image_filename: str
    class_id: int
    class_name: str
    kind: AnnotationKind
    geometry: Geometry

    @property
    def area(self) -> float:
        return self.geometry.area

    @property
    def tight_box(self) -> Box:
        return self.geometry.tight_box


@dataclass
class Dataset:
    """In-memory view a Check can iterate. Not frozen because derived
    lookups are memoized on first access."""

    images: List[Image]
    annotations: List[Annotation]
    categories: List[Category]
    # Filenames that were listed in the source folder/manifest but could not
    # be read. Kept as top-level metadata because there's no sensible
    # placeholder for a corrupt Image (no dimensions, no path).
    corrupt_filenames: List[str] = field(default_factory=list)
    # Optional split label per image filename. Populated only when callers
    # explicitly pass split info; checks that need it (leakage, purity)
    # short-circuit when missing.
    split_by_filename: Dict[str, str] = field(default_factory=dict)

    _image_by_filename: Optional[Dict[str, Image]] = field(
        default=None, init=False, repr=False
    )
    _annotations_by_image: Optional[Dict[str, List[Annotation]]] = field(
        default=None, init=False, repr=False
    )
    _annotations_by_class: Optional[Dict[int, List[Annotation]]] = field(
        default=None, init=False, repr=False
    )
    _kinds: Optional[FrozenSet[AnnotationKind]] = field(
        default=None, init=False, repr=False
    )

    @property
    def image_by_filename(self) -> Dict[str, Image]:
        if self._image_by_filename is None:
            self._image_by_filename = {img.filename: img for img in self.images}
        return self._image_by_filename

    @property
    def annotations_by_image(self) -> Dict[str, List[Annotation]]:
        if self._annotations_by_image is None:
            grouped: Dict[str, List[Annotation]] = {}
            for ann in self.annotations:
                grouped.setdefault(ann.image_filename, []).append(ann)
            self._annotations_by_image = grouped
        return self._annotations_by_image

    @property
    def annotations_by_class(self) -> Dict[int, List[Annotation]]:
        if self._annotations_by_class is None:
            grouped: Dict[int, List[Annotation]] = {}
            for ann in self.annotations:
                grouped.setdefault(ann.class_id, []).append(ann)
            self._annotations_by_class = grouped
        return self._annotations_by_class

    @property
    def kinds(self) -> FrozenSet[AnnotationKind]:
        if self._kinds is None:
            self._kinds = frozenset(ann.kind for ann in self.annotations)
        return self._kinds

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_annotations(self) -> int:
        return len(self.annotations)

    @property
    def num_classes(self) -> int:
        return len(self.categories)
