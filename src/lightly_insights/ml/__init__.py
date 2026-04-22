"""Optional ML helpers — pretrained-model label proposers.

Everything here requires the ``[ml]`` extra (installs ``ultralytics``).
The module imports succeed without it; the functions raise a clear
ImportError only when called.

Proposers return ``List[Annotation]`` with ``source`` tagged
``"proposal:<model-name>"``. Merge them with existing annotations:

    proposals = propose_with_yolo(paths, next_ann_id=max(a.annotation_id for a in ds.annotations) + 1)
    ds = Dataset(..., annotations=[*ds.annotations, *proposals])

Then ``missing_label_proposal`` (bundled check) flags proposals that have
no matching non-proposal annotation — these are your candidate missing
labels.
"""
from lightly_insights.ml.yolo_proposer import (  # noqa: F401
    PROPOSAL_SOURCE_PREFIX,
    is_proposal_source,
    propose_with_yolo,
)

__all__ = [
    "PROPOSAL_SOURCE_PREFIX",
    "is_proposal_source",
    "propose_with_yolo",
]
