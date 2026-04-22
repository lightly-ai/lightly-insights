"""Built-in checks.

Importing this package registers every bundled check. Custom checks can
live in user code and be registered via ``@register_check``; they'll show
up in ``list_checks()`` and be picked up by ``run_all``.
"""
# Side-effect imports register checks with the global registry.
from lightly_insights.checks import (  # noqa: F401
    background_scarcity,
    class_conflict,
    confidence_class_bias,
    confidence_low_pass,
    confidence_size_mismatch,
    corrupt_images,
    duplicate_annotation,
    geometric_validity,
    mask_fragmentation,
    missing_label_proposal,
    multi_source_disagreement,
    polygon_axis_aligned,
    polygon_mask_consistency,
    polygon_self_intersect,
    same_class_overlap,
    shape_outlier,
    split_leakage,
    split_purity,
    starved_class,
)
