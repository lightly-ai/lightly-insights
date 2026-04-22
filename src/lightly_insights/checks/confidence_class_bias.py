"""Per-class confidence bias: the autolabeler is more sure of some classes than others.

An autolabeler trained heavily on cars will return high confidences for
cars and mediocre confidences for bicycles. The bicycles aren't
necessarily wrong — but they need review priority. Surface the
discrepancy so reviewers know where to look.

We flag when the spread between the most-confident and least-confident
class means exceeds ``SPREAD_WARN`` and there's enough data per class.
"""
from __future__ import annotations

from statistics import mean
from typing import List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding, Severity

MIN_SAMPLES_PER_CLASS = 30
SPREAD_WARN = 0.2  # absolute difference in mean confidence between classes


@register_check
class ConfidenceClassBiasCheck(Check):
    check_id = "confidence_class_bias"
    title = "Autolabeler confidence varies by class"
    category = "balance"
    supported_kinds = frozenset(
        {AnnotationKind.BOX, AnnotationKind.POLYGON, AnnotationKind.MASK}
    )

    def applies_to(self, dataset: Dataset) -> bool:
        return any(a.confidence is not None for a in dataset.annotations)

    def run(self, dataset: Dataset) -> List[Finding]:
        by_class = dataset.annotations_by_class
        class_means: List = []  # (mean, name, count)
        for cat in dataset.categories:
            scored = [
                a.confidence for a in by_class.get(cat.id, [])
                if a.confidence is not None
            ]
            if len(scored) < MIN_SAMPLES_PER_CLASS:
                continue
            class_means.append((mean(scored), cat.name, len(scored)))
        if len(class_means) < 2:
            return []

        class_means.sort()
        worst_mean, worst_name, worst_n = class_means[0]
        best_mean, best_name, best_n = class_means[-1]
        spread = best_mean - worst_mean
        if spread < SPREAD_WARN:
            return []
        return [
            Finding(
                check_id=self.check_id,
                severity=Severity.MEDIUM,
                category=self.category,
                title=(
                    f"Autolabel confidence varies by {spread:.2f} across classes"
                ),
                detail=(
                    f"Mean confidence is {best_mean:.2f} for class "
                    f"'{best_name}' ({best_n} samples) vs {worst_mean:.2f} "
                    f"for class '{worst_name}' ({worst_n} samples). The "
                    "low-confidence class is a review priority — the "
                    "autolabeler is less sure of it and more likely to err."
                ),
                action=(
                    f"Prioritize human review of '{worst_name}' "
                    "annotations, or retrain the autolabeler with more "
                    "samples of that class."
                ),
                evidence={
                    "best_class": best_name,
                    "best_mean_confidence": round(best_mean, 3),
                    "worst_class": worst_name,
                    "worst_mean_confidence": round(worst_mean, 3),
                    "spread": round(spread, 3),
                },
            )
        ]
