"""The ``Check`` abstract base and the global registry.

Usage pattern:

    @register_check
    class MyCheck(Check):
        check_id = "my_check"
        title = "My check"
        category = "annotation"
        supported_kinds = frozenset({AnnotationKind.BOX, AnnotationKind.POLYGON})

        def run(self, dataset: Dataset) -> list[Finding]:
            ...

A check that doesn't declare ``supported_kinds`` (or declares an empty set)
applies to any dataset. The runner filters based on ``dataset.kinds`` before
invoking ``run()``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, FrozenSet, List, Type

from lightly_insights.core.dataset import AnnotationKind, Dataset
from lightly_insights.core.finding import Finding

# Module-level registry. Populated by @register_check as modules are imported.
_CHECK_REGISTRY: "dict[str, Type[Check]]" = {}


class Check(ABC):
    """Abstract base. Subclasses declare class-level metadata and a
    single ``run`` method that returns findings for the supplied dataset."""

    check_id: ClassVar[str]
    title: ClassVar[str]
    category: ClassVar[str] = "quality"
    # Empty frozenset = applies to any dataset regardless of annotation kind.
    supported_kinds: ClassVar[FrozenSet[AnnotationKind]] = frozenset()

    def applies_to(self, dataset: Dataset) -> bool:
        """Does this dataset carry annotation kinds this check understands?

        Checks that don't care about annotations (image-level checks) should
        override to return True unconditionally.
        """
        if not self.supported_kinds:
            return True
        return bool(dataset.kinds & self.supported_kinds)

    @abstractmethod
    def run(self, dataset: Dataset) -> List[Finding]:
        raise NotImplementedError


def register_check(cls: Type[Check]) -> Type[Check]:
    """Class decorator that adds the check to the global registry."""
    if not getattr(cls, "check_id", None):
        raise ValueError(f"Check {cls.__name__} must define a check_id class attribute.")
    if cls.check_id in _CHECK_REGISTRY:
        # Later definitions win — useful for test overrides.
        pass
    _CHECK_REGISTRY[cls.check_id] = cls
    return cls
