"""Classes with too few annotations to learn reliably.

Works on any annotation kind — counts are geometry-independent. Emits one
Finding per starved class so reporters can render them as a prioritized
list.
"""
from __future__ import annotations

from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity

MIN_OBJECTS_PER_CLASS = 30


@register_check
class StarvedClassCheck(Check):
    check_id = "starved_class"
    title = "Under-represented class"
    category = "balance"
    # No supported_kinds restriction — counts work across box/polygon/mask.

    def run(self, dataset: Dataset) -> List[Finding]:
        findings: List[Finding] = []
        by_class = dataset.annotations_by_class
        for cat in dataset.categories:
            count = len(by_class.get(cat.id, []))
            if count == 0:
                # Orphan class — declared but not used.
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        severity=Severity.HIGH,
                        category=self.category,
                        title=f"Class '{cat.name}' has no annotations",
                        detail=(
                            f"Class '{cat.name}' (id {cat.id}) is declared in "
                            "the schema but appears in zero annotations."
                        ),
                        action="Remove the class from the schema or add samples.",
                        evidence={"class_id": cat.id, "class_name": cat.name, "count": 0},
                    )
                )
            elif count < MIN_OBJECTS_PER_CLASS:
                findings.append(
                    Finding(
                        check_id=self.check_id,
                        severity=Severity.HIGH,
                        category=self.category,
                        title=f"Class '{cat.name}' is under-represented",
                        detail=(
                            f"Class '{cat.name}' has only {count} annotations "
                            f"(minimum for reliable learning: {MIN_OBJECTS_PER_CLASS})."
                        ),
                        action="Collect more samples of this class or merge into a parent class.",
                        evidence={
                            "class_id": cat.id,
                            "class_name": cat.name,
                            "count": count,
                            "threshold": MIN_OBJECTS_PER_CLASS,
                        },
                    )
                )
        return findings
