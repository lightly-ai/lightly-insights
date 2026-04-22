# Bundled checks

21 checks registered. Each one emits `Finding` objects with a severity, category, and a concrete action string.

## Pipeline integrity (CRITICAL severity)

| `check_id` | What it catches |
|---|---|
| `corrupt_images` | Images listed in the dataset that PIL can't open. |
| `degenerate_annotation` | Zero-area boxes, polygons with <3 points, all-zero masks. |
| `out_of_bounds_annotation` | Annotations extending past the image's width or height. |
| `mask_image_size_mismatch` | Mask shape ≠ corresponding image dimensions. Silent disaster at training time. |

## Annotation quality

| `check_id` | What it catches | Scope |
|---|---|---|
| `duplicate_annotation` | Same-class annotations with IoU ≥ 0.9 (near-identical). | box / polygon / mask |
| `same_class_overlap` | Same-class boxes where one covers ≥ 70 % of the other's area (touching objects merged). | box / polygon / mask |
| `class_conflict` | Different-class boxes overlapping with IoU ≥ 0.5 (likely mislabel). | box / polygon / mask |
| `shape_outlier` | Per class: annotations >5 MADs from the class median in area or aspect. | box / polygon / mask |
| `polygon_self_intersect` | Polygons whose edges cross themselves (bowtie shapes). | polygon only |
| `polygon_axis_aligned` | Polygons with >85 % horizontal/vertical edges (rectangle-trace labeling). | polygon only |
| `polygon_mask_consistency` | Paired polygon + mask with rasterized IoU < 0.9 (one's stale). | polygon + mask |
| `mask_fragmentation` | Masks with > 3 significant connected components. | mask only |
| `multi_source_disagreement` | Two `source` values produce different classes or one misses what the other saw. | box / polygon / mask |
| `missing_label_proposal` | `source="proposal:*"` annotations (from a pretrained model) with no matching real annotation. | box / polygon / mask |

## Balance & data coverage

| `check_id` | What it catches |
|---|---|
| `starved_class` | Orphan classes (0 annotations) + classes with < 30 annotations. |
| `background_scarcity` | Datasets with < 5 % zero-annotation images. |
| `split_purity` | Class distribution drift across splits (χ² p < 0.05). |
| `split_leakage` | Near-duplicate images spanning ≥ 2 splits (needs `imagehash` extra). |

## Autolabel-specific (active when `confidence` is set)

| `check_id` | What it catches |
|---|---|
| `confidence_low_pass` | Autolabels below 0.3 confidence (aggregated per class). |
| `confidence_size_mismatch` | Confidence ≥ 0.9 on annotations covering < 0.5 % of the image (classic over-confident-tiny failure). |
| `confidence_class_bias` | Spread ≥ 0.2 between best- and worst-confidence class means. |

## Tuning thresholds

Most checks expose their threshold as a module constant at the top of their file:

```python
# checks/starved_class.py
MIN_OBJECTS_PER_CLASS = 30

# checks/shape_outlier.py
MAD_THRESHOLD = 5.0
MIN_CLASS_SAMPLES = 20

# checks/confidence_low_pass.py
LOW_CONFIDENCE_THRESHOLD = 0.3
```

Edit those directly to match your domain. The checks don't take per-run config yet — opinionated defaults + monkey-patch is the current extensibility contract.

## Applies-to logic

Each check declares when it should run. The default `Check.applies_to`
returns `True` unconditionally; checks that need specific data override it:

- Checks with an explicit `supported_kinds` frozenset apply when
  `dataset.kinds & check.supported_kinds` is non-empty.
- Checks with an empty `supported_kinds` (image-level checks like
  `corrupt_images`, `background_scarcity`, `starved_class`,
  `split_purity`, `split_leakage`, `confidence_class_bias`) apply
  regardless of annotation kinds.
- `polygon_mask_consistency` applies only when both polygon AND mask
  annotations are present.
- `confidence_*` checks apply only when at least one annotation carries
  a confidence.
- `split_purity` / `split_leakage` apply only when
  `dataset.split_by_filename` has ≥ 2 splits.
- `multi_source_disagreement` applies only when ≥ 2 distinct `source`
  values are present.
- `missing_label_proposal` applies only when any annotation has a
  `source` starting with `"proposal:"`.

`run_all` filters automatically via `check.applies_to(dataset)`; your
code doesn't have to reason about which checks are relevant.

## Severity semantics

Lower numeric value = more urgent.

| Constant | Value | Meaning |
|---|---|---|
| `Severity.CRITICAL` | 10 | Pipeline-breaking; fix before analyzing anything else. |
| `Severity.HIGH` | 30 | Likely wrong labels; high-value review target. |
| `Severity.MEDIUM` | 50 | Probably wrong or bias signal; review when you have time. |
| `Severity.LOW` | 70 | Informational noise; batch-triage. |
| `Severity.INFO` | 90 | Config/meta messages. |

Intermediate integer values are allowed (e.g. `severity=25` for "between CRITICAL and HIGH"); the summary-strip bucketing in `findings_present._severity_buckets` covers the full 0-100 space with no gaps. (The letter-grade logic in `present._letter_grade` is unrelated — it grades a 0-100 quality *score*, not a severity.)
