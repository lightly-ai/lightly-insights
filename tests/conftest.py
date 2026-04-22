"""Test infrastructure.

The check registry is a global dict populated by ``@register_check`` at
module import time. Tests that temporarily register ad-hoc checks to
exercise the registry surface (e.g. ``test_run_all_isolates_crashing_check``)
used to leak those entries into every subsequent test's ``run_all()`` call,
which silently flipped depending on test ordering. The fixture below
snapshots the registry around every test so additions are rolled back.
"""
from __future__ import annotations

import pytest

from lightly_insights.core.check import _CHECK_REGISTRY


@pytest.fixture(autouse=True)
def _restore_check_registry():
    """Snapshot and restore the global check registry around each test."""
    snapshot = dict(_CHECK_REGISTRY)
    yield
    # Remove any ids the test added.
    for key in list(_CHECK_REGISTRY.keys()):
        if key not in snapshot:
            del _CHECK_REGISTRY[key]
    # Restore any replaced entries to their original class.
    for key, cls in snapshot.items():
        _CHECK_REGISTRY[key] = cls
