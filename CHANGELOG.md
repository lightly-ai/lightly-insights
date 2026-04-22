# Changelog

## Unreleased — check/finding architecture

Reframed the tool around a check/finding/review-queue framework so downstream
consumers (LightlyStudio, CI gates, custom reporters) get structured output
instead of scraping HTML. The legacy `analyze` → `present` pipeline is
untouched and still ships.

### New

- **Core framework** (`core/`)
  - `Dataset` / `Annotation` / `AnnotationKind` / `Category` / `Image` —
    unified in-memory representation of an OD / segmentation dataset.
  - `Box` / `Polygon` / `Mask` geometry primitives with a common API
    (`area`, `tight_box`, IoU, polygon self-intersection, axis-aligned
    edge fraction, pure-numpy mask connected components).
  - `Finding` + `Severity` immutable records with `to_dict()` for JSON
    serialization.
  - `Check` ABC + `@register_check` decorator; per-check
    `supported_kinds` drives automatic filtering by geometry.
  - `run_all(dataset, strict=False)` with per-check exception isolation —
    a single buggy check can't take down a whole analysis run.
  - `build_review_queue(findings, dataset, max_items)` ranks entries by
    `severity × (1 + uncertainty)`, emits `ReviewItem`s.
  - `export_review_queue_csv` side-car writer.
  - `core/adapter.py::build_dataset` bridges the legacy `analyze.*`
    results into the new framework.
  - `core/overview_plots.py` — 5 curated plots (class composition,
    confidence distribution, spatial heatmap, findings-by-check,
    cumulative review burden).
  - `core/drilldown.py` — per-finding thumbnail panels with class
    exemplars drawn alongside.
  - `core/attribution.py` — per-source finding aggregation (which
    autolabeler / annotator is the worst offender).

- **Autolabel support**
  - `Annotation.confidence` and `Annotation.source` fields.
  - `lightly_insights.ml` optional module (requires `[ml]` extra):
    `propose_with_yolo(image_paths)` emits annotations tagged
    `source="proposal:<model>"`.
  - `[ml]` extra pulls in `ultralytics`; `[near-duplicates]` extra pulls
    in `imagehash`. Both are optional.

- **Bundled checks** (21 total)
  - Pipeline integrity: `corrupt_images`, `degenerate_annotation`,
    `out_of_bounds_annotation`, `mask_image_size_mismatch`.
  - Annotation quality: `duplicate_annotation`, `same_class_overlap`,
    `class_conflict`, `shape_outlier`, `polygon_self_intersect`,
    `polygon_axis_aligned`, `polygon_mask_consistency`,
    `mask_fragmentation`, `multi_source_disagreement`,
    `missing_label_proposal`.
  - Balance: `starved_class`, `background_scarcity`, `split_purity`
    (pure-Python χ² test), `split_leakage` (perceptual-hash union-find
    across splits).
  - Autolabel-specific: `confidence_low_pass`, `confidence_size_mismatch`,
    `confidence_class_bias`.

- **Slim findings-first report** (`findings_present.py`)
  - New HTML renderer consumes `Dataset` + `findings` + `review_queue`
    directly, no dependency on legacy analyses.
  - Side-car exports: `review_queue.csv`, `findings.json`, plus static
    Bootstrap assets.
  - Layout: severity summary + workload plots + attribution + review
    queue (clickable ranks) + dataset-at-a-glance plots + collapsible
    findings by category + drill-down panels.

- **Documentation**
  - `docs/architecture.md` — framework overview.
  - `docs/checks.md` — reference for all 23 bundled checks.
  - `docs/metrics_ideas.md` — proposed but not yet implemented metrics,
    ranked by expected leverage.

### Fixed (from an internal correctness review)

- `review_queue.build_review_queue` — sort tiebreak no longer conflates
  `annotation_id=0` with `None`.
- `confidence_low_pass` — aggregates one finding per class instead of
  emitting one per affected annotation; avoids exploding the review
  queue on large autolabel sets.
- `findings_present._severity_buckets` — ranges cover the full 0-100
  severity space; custom severities like 25/35/45 are no longer
  silently dropped from the summary strip.
- `core/drilldown._draw_annotation` — PIL imports hoisted to module
  scope so a broken PIL install raises ImportError at import time
  instead of NameError in an except clause.
- `register_check` — duplicate `check_id` now logs a warning before
  overwriting (previously silent).
- `core/adapter.build_dataset` — `Image.path` is `None` for filenames
  whose file isn't actually on disk; downstream checks skip a stat per
  image.
- `core/drilldown.build_drilldowns` — precomputes `annotation_id → ann`
  once; previously fell back to O(n) scan for each non-contiguous id
  (common when ML proposals are merged in).

### Backwards compatibility

- `lightly_insights.analyze.*` and `lightly_insights.present.*`
  signatures are unchanged. Existing scripts continue to work.
- `lightly_insights.compare` remains the recommended way to diff two
  runs.

## Earlier releases

See git log for the pre-framework work (scorecard, training advice,
per-class tabs, anchor recommendations, class-imbalance strip,
duplicate-annotation detection, heatmap overlay, etc.).
