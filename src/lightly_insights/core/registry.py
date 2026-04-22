"""Public registry helpers: list_checks, run_all.

Kept separate from ``check.py`` so importing ``Check`` / ``register_check``
doesn't pull in the check implementations (no import cycle).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from lightly_insights.core.check import _CHECK_REGISTRY, Check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding


def list_checks() -> Dict[str, Type[Check]]:
    """Copy of the global check registry keyed by check_id."""
    return dict(_CHECK_REGISTRY)


def run_all(
    dataset: Dataset,
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[Finding]:
    """Run every registered check against the dataset, sorted by severity.

    ``only`` / ``exclude`` filter by check_id. Checks whose ``applies_to``
    returns False for this dataset are skipped silently.
    """
    findings: List[Finding] = []
    for check_id, cls in _CHECK_REGISTRY.items():
        if only is not None and check_id not in only:
            continue
        if exclude is not None and check_id in exclude:
            continue
        check = cls()
        if not check.applies_to(dataset):
            continue
        findings.extend(check.run(dataset))
    findings.sort(key=lambda f: (f.severity, f.check_id))
    return findings
