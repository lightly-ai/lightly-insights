"""Cross-split image leakage via perceptual hashing.

A near-duplicate image that appears in both train and val invalidates
the val metric — the model effectively saw the "held-out" sample. This
is one of the single most common ways ML teams ship inflated numbers.

Uses dHash (perceptual hash) via the optional ``imagehash`` package. The
check silently no-ops if ``imagehash`` isn't installed or if the dataset
doesn't have splits configured.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity

logger = logging.getLogger(__name__)

# Hamming-distance threshold on 8-bit dHash (out of 64 bits). Lower =
# stricter. 5 catches near-identical images without tripping on texture
# repeats in natural photos.
HAMMING_THRESHOLD = 5


@register_check
class SplitLeakageCheck(Check):
    check_id = "split_leakage"
    title = "Near-duplicate images across splits"
    category = "balance"

    def applies_to(self, dataset: Dataset) -> bool:
        if not dataset.split_by_filename:
            return False
        splits = set(dataset.split_by_filename.values())
        return len(splits) >= 2

    def run(self, dataset: Dataset) -> List[Finding]:
        try:
            import imagehash  # type: ignore[import]
            from PIL import Image as PILImage
        except ImportError:
            logger.info(
                "imagehash not installed; skipping split_leakage check. "
                "Install with the 'near-duplicates' extra to enable."
            )
            return []

        # Hash each image we have a path for.
        image_by_name = dataset.image_by_filename
        hashes: Dict[str, object] = {}
        for filename, split in dataset.split_by_filename.items():
            image = image_by_name.get(filename)
            if image is None or image.path is None or not image.path.exists():
                continue
            try:
                with PILImage.open(image.path) as img:
                    hashes[filename] = imagehash.dhash(img)
            except Exception as exc:  # noqa: BLE001 -- we want to skip any bad file
                logger.debug(f"Could not hash {filename}: {exc}")
                continue

        if len(hashes) < 2:
            return []

        # Union-find over near-duplicate pairs.
        names = sorted(hashes.keys())
        parent: Dict[str, str] = {n: n for n in names}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            parent[find(a)] = find(b)

        # O(n²) pairwise. Fine up to a few thousand images; LSH would be
        # needed beyond that.
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if hashes[names[i]] - hashes[names[j]] <= HAMMING_THRESHOLD:  # type: ignore[operator]
                    union(names[i], names[j])

        # Bucket by root and report groups that span multiple splits.
        groups: Dict[str, List[str]] = defaultdict(list)
        for n in names:
            groups[find(n)].append(n)

        findings: List[Finding] = []
        for members in groups.values():
            seen_splits = {dataset.split_by_filename[m] for m in members}
            if len(seen_splits) < 2:
                continue
            members_sorted = sorted(members)
            findings.append(
                Finding(
                    check_id=self.check_id,
                    severity=Severity.CRITICAL,
                    category=self.category,
                    title=f"Near-duplicate images span splits {sorted(seen_splits)}",
                    detail=(
                        f"{len(members_sorted)} visually-similar images "
                        f"appear in splits {sorted(seen_splits)}. The val/"
                        "test metric is measuring performance on samples "
                        "the model has effectively seen during training."
                    ),
                    action=(
                        "Move all members of each leak group into a "
                        "single split before measuring eval numbers."
                    ),
                    affected_images=members_sorted,
                    evidence={
                        "splits": sorted(seen_splits),
                        "group_size": len(members_sorted),
                    },
                )
            )
        return findings
