"""Public registry helpers: list_checks, run_all.

Kept separate from ``check.py`` so importing ``Check`` / ``register_check``
doesn't pull in the check implementations (no import cycle).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from lightly_insights.core.check import _CHECK_REGISTRY, Check
from lightly_insights.core.dataset import Dataset
from lightly_insights.core.finding import Finding, Severity

logger = logging.getLogger(__name__)


def list_checks() -> Dict[str, Type[Check]]:
    """Copy of the global check registry keyed by check_id."""
    return dict(_CHECK_REGISTRY)


def run_all(
    dataset: Dataset,
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    strict: bool = False,
) -> List[Finding]:
    """Run every registered check against the dataset, sorted by severity.

    ``only`` / ``exclude`` filter by check_id. Checks whose ``applies_to``
    returns False for this dataset are skipped silently.

    When a check crashes:
      - ``strict=False`` (default): swallow the exception, emit an INFO-
        severity Finding so the failure is visible in the report, and
        continue with the remaining checks. A single buggy check can't
        take down a whole analysis run — important for production use.
      - ``strict=True``: re-raise the original exception immediately. Use
        during development and in unit tests so bugs surface instead of
        being quietly annotated.
    """
    findings: List[Finding] = []
    # Iterate in a deterministic order so two runs on the same dataset yield
    # bit-for-bit identical findings.json. Relies on check_id being unique.
    for check_id, cls in sorted(_CHECK_REGISTRY.items()):
        if only is not None and check_id not in only:
            continue
        if exclude is not None and check_id in exclude:
            continue
        try:
            check = cls()
            if not check.applies_to(dataset):
                continue
            findings.extend(check.run(dataset))
        except Exception as exc:  # noqa: BLE001 -- we deliberately catch all
            if strict:
                raise
            logger.exception(f"Check '{check_id}' crashed: {exc}")
            findings.append(
                Finding(
                    check_id=check_id,
                    severity=Severity.INFO,
                    category="meta",
                    title=f"Check '{check_id}' crashed",
                    detail=(
                        f"{type(exc).__name__}: {exc}. The check was "
                        "skipped; other checks ran normally."
                    ),
                    action=(
                        "Report as a bug with the dataset shape that "
                        "triggered this crash."
                    ),
                    evidence={"exception": f"{type(exc).__name__}: {exc}"},
                )
            )
    findings.sort(key=lambda f: (f.severity, f.check_id))
    return findings
