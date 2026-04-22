"""Built-in checks.

Importing this package registers every bundled check. Custom checks can
live in user code and be registered via ``@register_check``; they'll show
up in ``list_checks()`` and be picked up by ``run_all``.
"""
# Side-effect imports register checks with the global registry.
from lightly_insights.checks import (  # noqa: F401
    class_conflict,
    corrupt_images,
    duplicate_annotation,
    mask_fragmentation,
    polygon_axis_aligned,
    polygon_self_intersect,
    shape_outlier,
    starved_class,
)
