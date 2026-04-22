"""Geometry primitives: Box, Polygon, Mask.

All three expose a minimum common API (``area``, ``tight_box``) so checks
can operate regardless of annotation kind. Checks that need kind-specific
math (e.g. self-intersection) still have access to the underlying storage.

Boxes store absolute pixel coordinates in (xmin, ymin, xmax, ymax). Polygons
store a closed list of (x, y) vertices. Masks wrap a boolean NDArray of
shape (H, W); shape is enforced lazily to keep the dataclass cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / height. Returns 0.0 if height is zero."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def tight_box(self) -> "Box":
        return self

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def iou(self, other: "Box") -> float:
        x1 = max(self.xmin, other.xmin)
        y1 = max(self.ymin, other.ymin)
        x2 = min(self.xmax, other.xmax)
        y2 = min(self.ymax, other.ymax)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class Polygon:
    # A single closed polygon. `points` should not repeat the first point at
    # the end; closure is implicit.
    points: Tuple[Tuple[float, float], ...]

    @property
    def area(self) -> float:
        """Shoelace formula. Always non-negative."""
        n = len(self.points)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    @property
    def perimeter(self) -> float:
        n = len(self.points)
        if n < 2:
            return 0.0
        total = 0.0
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total

    @property
    def tight_box(self) -> Box:
        if not self.points:
            return Box(0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Box(xmin=min(xs), ymin=min(ys), xmax=max(xs), ymax=max(ys))


@dataclass(frozen=True)
class Mask:
    """Binary mask, shape (H, W), dtype bool. Coordinates are pixel-space."""

    array: NDArray[np.bool_]

    @property
    def area(self) -> float:
        return float(self.array.sum())

    @property
    def tight_box(self) -> Box:
        if self.array.size == 0 or not self.array.any():
            return Box(0.0, 0.0, 0.0, 0.0)
        ys, xs = np.nonzero(self.array)
        return Box(
            xmin=float(xs.min()),
            ymin=float(ys.min()),
            # +1 because numpy indices are inclusive-exclusive for slicing.
            xmax=float(xs.max() + 1),
            ymax=float(ys.max() + 1),
        )

    def connected_components(self) -> int:
        """Number of 4-connected foreground components. Pure-numpy BFS."""
        if self.array.size == 0 or not self.array.any():
            return 0
        visited = np.zeros_like(self.array, dtype=bool)
        arr = self.array
        h, w = arr.shape
        components = 0
        for y in range(h):
            for x in range(w):
                if not arr[y, x] or visited[y, x]:
                    continue
                components += 1
                # BFS
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx] or not arr[cy, cx]:
                        continue
                    visited[cy, cx] = True
                    if cy + 1 < h and not visited[cy + 1, cx]:
                        stack.append((cy + 1, cx))
                    if cy > 0 and not visited[cy - 1, cx]:
                        stack.append((cy - 1, cx))
                    if cx + 1 < w and not visited[cy, cx + 1]:
                        stack.append((cy, cx + 1))
                    if cx > 0 and not visited[cy, cx - 1]:
                        stack.append((cy, cx - 1))
        return components
