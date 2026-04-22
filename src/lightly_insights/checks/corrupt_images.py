"""Flag images that were listed in the dataset but couldn't be read."""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity


@register_check
class CorruptImagesCheck(Check):
    check_id = "corrupt_images"
    title = "Corrupt / unreadable images"
    category = "quality"
    # Image-level; works on any dataset.

    def run(self, dataset: Dataset) -> List[Finding]:
        corrupt = list(dataset.corrupt_filenames)
        if not corrupt:
            return []
        return [
            Finding(
                check_id=self.check_id,
                severity=Severity.CRITICAL,
                category=self.category,
                title=self.title,
                detail=(
                    f"{len(corrupt)} image(s) could not be opened by PIL. "
                    "They will silently disappear from any subsequent analysis."
                ),
                action="Remove or replace the listed files before training.",
                affected_images=sorted(corrupt),
                evidence={"count": len(corrupt)},
            )
        ]
