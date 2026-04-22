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

    def self_intersects(self) -> bool:
        """True if any two non-adjacent edges cross.

        O(n^2) sweep — fine for the annotation scale we expect (polygons
        rarely exceed a few hundred vertices). For massive polygons a
        Bentley-Ottmann sweep would be O(n log n), but the per-dataset
        count makes that premature.
        """
        n = len(self.points)
        if n < 4:
            return False
        edges = [
            (self.points[i], self.points[(i + 1) % n]) for i in range(n)
        ]
        for i in range(n):
            for j in range(i + 1, n):
                # Skip adjacent edges (they share an endpoint by construction).
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if _segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                    return True
        return False

    def axis_aligned_edge_fraction(self, angle_tolerance_deg: float = 2.0) -> float:
        """Fraction of edges that are horizontal or vertical within tolerance.

        High values (>0.6) are a red flag: labelers who drew axis-aligned
        rectangles instead of tracing the object outline.
        """
        import math

        n = len(self.points)
        if n < 3:
            return 0.0
        tol = math.tan(math.radians(angle_tolerance_deg))
        axis = 0
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx == 0 and dy == 0:
                continue  # degenerate edge, ignore
            # Horizontal: dy/dx small. Vertical: dx/dy small.
            if dx > 0 and dy / dx <= tol:
                axis += 1
            elif dy > 0 and dx / dy <= tol:
                axis += 1
        # Non-degenerate edges only; denominator re-counted above.
        denom = sum(
            1
            for i in range(n)
            if self.points[i] != self.points[(i + 1) % n]
        )
        return axis / denom if denom > 0 else 0.0


def _orientation(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> int:
    """0 = collinear, >0 = counter-clockwise, <0 = clockwise. Uses the sign
    of the cross product of PQ and PR."""
    val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
    """True if q lies on segment pr, given that p, q, r are collinear."""
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def _segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Do segment p1p2 and segment p3p4 intersect? Standard CCW test."""
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    # Collinear special cases.
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


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
