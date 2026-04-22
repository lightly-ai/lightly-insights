"""Finding: a single issue emitted by a Check.

Lower ``severity`` = more urgent. The integer makes sort/rank deterministic
and lets us tune without recompiling consumers. The ``category`` is
a coarse bucket (used for grouping in reports) and is independent of
severity. ``evidence`` is free-form JSON-ish data reporters can splice into
their own output; keep it small and primitive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List


class Severity(IntEnum):
    """Coarse severity buckets. Use custom integer values for fine-grained
    ranking inside a bucket.
    """

    CRITICAL = 10
    HIGH = 30
    MEDIUM = 50
    LOW = 70
    INFO = 90


@dataclass(frozen=True)
class Finding:
    check_id: str
    severity: int
    category: str  # e.g. "annotation", "quality", "balance", "training"
    title: str  # short, human-readable
    detail: str  # longer explanation of what was found
    action: str  # one-line suggested fix
    affected_images: List[str] = field(default_factory=list)
    affected_annotations: List[int] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for exports. Evidence fields must already be
        JSON-encodable; that's the caller's responsibility."""
        return {
            "check_id": self.check_id,
            "severity": int(self.severity),
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
            "affected_images": list(self.affected_images),
            "affected_annotations": list(self.affected_annotations),
            "evidence": dict(self.evidence),
        }
