"""Class distribution drift across splits (χ² test).

If the class distribution in val/test differs materially from train, your
eval metrics are biased — you're measuring accuracy on a population the
model wasn't trained for.

Uses a pure-Python chi-square computation (no scipy dep). Only runs when
``Dataset.split_by_filename`` is populated and carries at least two
distinct split names.
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import Dict, List

from lightly_insights.core.check import Check, register_check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity

# p-value threshold. 0.05 is the classic statistical cutoff.
P_VALUE_THRESHOLD = 0.05


@register_check
class SplitPurityCheck(Check):
    check_id = "split_purity"
    title = "Class distribution drifts across splits"
    category = "balance"

    def applies_to(self, dataset: Dataset) -> bool:
        if not dataset.split_by_filename:
            return False
        splits = set(dataset.split_by_filename.values())
        return len(splits) >= 2

    def run(self, dataset: Dataset) -> List[Finding]:
        # Build a (split, class) count table.
        splits = sorted(set(dataset.split_by_filename.values()))
        class_ids = sorted({c.id for c in dataset.categories})
        if not class_ids:
            return []
        class_name = {c.id: c.name for c in dataset.categories}

        table: Dict[str, Dict[int, int]] = {s: {cid: 0 for cid in class_ids} for s in splits}
        for ann in dataset.annotations:
            split = dataset.split_by_filename.get(ann.image_filename)
            if split is None:
                continue
            if ann.class_id in table[split]:
                table[split][ann.class_id] += 1

        # Compute χ² against the null "every split shares the same class
        # distribution". Skip when any margin is zero.
        totals_per_split = {s: sum(table[s].values()) for s in splits}
        totals_per_class = {cid: sum(table[s][cid] for s in splits) for cid in class_ids}
        grand_total = sum(totals_per_split.values())
        if grand_total == 0:
            return []

        chi2 = 0.0
        dof = 0
        for s in splits:
            for cid in class_ids:
                expected = totals_per_split[s] * totals_per_class[cid] / grand_total
                if expected == 0:
                    continue
                observed = table[s][cid]
                chi2 += (observed - expected) ** 2 / expected
                dof += 1
        # DoF = (rows - 1) * (cols - 1); the loop counted rows*cols non-zero cells.
        true_dof = (len(splits) - 1) * (len(class_ids) - 1)
        if true_dof <= 0:
            return []
        p_value = _chi2_sf(chi2, true_dof)
        if p_value >= P_VALUE_THRESHOLD:
            return []

        # Report the most skewed (split, class) cells for context.
        skew: List = []
        for s in splits:
            for cid in class_ids:
                expected = totals_per_split[s] * totals_per_class[cid] / grand_total
                observed = table[s][cid]
                if expected > 0:
                    skew.append(
                        (
                            (observed - expected) / expected ** 0.5,
                            s,
                            class_name.get(cid, str(cid)),
                            observed,
                            expected,
                        )
                    )
        skew.sort(key=lambda t: -abs(t[0]))
        top_skew = skew[:3]
        examples = "; ".join(
            f"{split}/{cls} obs={obs} vs expected={exp:.1f}"
            for _, split, cls, obs, exp in top_skew
        )

        return [
            Finding(
                check_id=self.check_id,
                severity=Severity.HIGH,
                category=self.category,
                title="Splits have different class distributions",
                detail=(
                    f"χ² = {chi2:.1f} (dof {true_dof}, p = {p_value:.4f}) "
                    "rejects the null hypothesis that splits share the "
                    f"same class distribution. Most skewed cells: {examples}."
                ),
                action=(
                    "Re-stratify the splits so each class is represented "
                    "proportionally, or weight the eval metric accordingly."
                ),
                evidence={
                    "chi2": round(chi2, 3),
                    "dof": true_dof,
                    "p_value": round(p_value, 6),
                    "num_splits": len(splits),
                    "num_classes": len(class_ids),
                },
            )
        ]


def _chi2_sf(x: float, dof: int) -> float:
    """Survival function (1 - CDF) for the chi-square distribution.

    Regularized upper incomplete gamma via a series/continued-fraction
    combination. Good enough for dof up to a few hundred without scipy.
    """
    if x <= 0:
        return 1.0
    a = dof / 2.0
    z = x / 2.0
    # Small z: use series. Large z: use continued fraction.
    if z < a + 1.0:
        return 1.0 - _gammap_series(a, z)
    return _gammaq_cf(a, z)


def _log_gamma(z: float) -> float:
    """Stirling-based Lanczos approximation."""
    # Lanczos coefficients (g=7, n=9).
    from math import log, pi, sin

    if z < 0.5:
        return log(pi / sin(pi * z)) - _log_gamma(1 - z)
    z -= 1
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    g = 7
    x = coeffs[0]
    for i in range(1, 9):
        x += coeffs[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * log(2 * pi) + (z + 0.5) * log(t) - t + log(x)


def _gammap_series(a: float, z: float) -> float:
    from math import exp, log

    if z == 0:
        return 0.0
    # Series: exp(-z + a*ln z - log_gamma(a)) * sum_{n=0..} z^n / (a*(a+1)*...*(a+n))
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(200):
        ap += 1
        term *= z / ap
        total += term
        if abs(term) < abs(total) * 1e-12:
            break
    return total * exp(-z + a * log(z) - _log_gamma(a))


def _gammaq_cf(a: float, z: float) -> float:
    from math import exp, log

    # Continued-fraction expansion for Q(a, z).
    b = z + 1 - a
    c = 1e30
    d = 1 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return exp(-z + a * log(z) - _log_gamma(a)) * h
