# Architecture

Two layers coexist in this codebase:

1. **Legacy pipeline** (`analyze.py`, `present.py`, `plots.py`, HTML template) — the original image-folder → HTML-report flow. Still fully functional.
2. **Check/finding framework** (`core/*`, `checks/*`, `findings_present.py`) — the forward-facing, library-first architecture. This is what LightlyStudio consumes.

New work lands in the framework. The legacy pipeline is frozen for backwards compatibility.

## Check/finding framework — high-level flow

```
┌─────────────────────────┐
│   Dataset               │  images + annotations + categories + (optional)
│   (core/dataset.py)     │  corrupt_filenames, split_by_filename
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   run_all(dataset)      │  iterates every @register_check class
│   (core/registry.py)    │  with per-check exception isolation
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   List[Finding]         │  (check_id, severity, category,
│   (core/finding.py)     │   title, detail, action, evidence)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│   build_review_queue(findings, dataset)     │  ranks by
│   (core/review_queue.py)                    │  severity × uncertainty
└──────────┬──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────┐
│   create_findings_report(                   │  slim HTML + review_queue.csv
│     output_folder, dataset,                 │  + findings.json + 5 plots
│     findings, review_queue)                 │  + drill-down panels
│   (findings_present.py)                    │
└─────────────────────────────────────────────┘
```

## Core types

### `Dataset`
Single source of truth. Carries images, annotations, categories, and optional metadata (corrupt filenames, split mapping). Derived views (`annotations_by_image`, `annotations_by_class`, `image_by_filename`, `kinds`) are memoized on first access.

### `Annotation`
- `annotation_id: int` — stable reference used by Findings
- `kind: AnnotationKind` — `BOX | POLYGON | MASK`
- `geometry: Box | Polygon | Mask` — uniform interface: every kind exposes `area` and `tight_box`
- `confidence: Optional[float]` — autolabel score when relevant
- `source: Optional[str]` — who produced it (`"human"`, `"yolo-v11"`, `"sam"`, `"proposal:yolov8n"`, …)

### `Finding`
Immutable result record. Contains everything a reporter needs to render one issue:
- `check_id`, `severity`, `category`, `title`, `detail`, `action`
- `affected_images: List[str]`, `affected_annotations: List[int]`
- `evidence: Dict[str, Any]` — check-specific metadata for drill-downs

### `Check`
Abstract base. Concrete checks subclass `Check`, declare:
- `check_id: str`, `title: str`, `category: str`
- `supported_kinds: FrozenSet[AnnotationKind]` — empty means "any kind"
- `applies_to(dataset) -> bool` — override when the check needs specific data (e.g. splits, confidences)
- `run(dataset) -> List[Finding]` — the actual logic

Register with `@register_check`. The global registry lives in `core/check.py`; lookups go through `core/registry.py`.

## Geometry primitives

All three geometries (`Box`, `Polygon`, `Mask`) expose the same minimum API:

| Property / method | Box | Polygon | Mask |
|---|---|---|---|
| `area` | w × h | shoelace | `sum()` |
| `tight_box` | self | AABB of points | nonzero AABB |
| `aspect_ratio` | w / h | `tight_box.aspect_ratio` | `tight_box.aspect_ratio` |
| Self-intersection | — | CCW sweep | — |
| Connected components | — | — | pure-numpy BFS |

This lets cross-geometry checks operate without branching.

## Adapters

### `core/adapter.py::build_dataset`
Builds a `Dataset` from the legacy `ImageAnalysis` + `ObjectDetectionAnalysis` objects plus an optional labelformat `ObjectDetectionInput`. Used to run the check framework on top of the existing analyze pipeline.

### `core/attribution.py::compute_attribution`
Buckets findings by annotation `source`. Emits ranked `AttributionItem`s for the "Findings by source" report section.

### `core/drilldown.py::build_drilldowns`
Pre-renders annotated thumbnails (flagged + class exemplar) for the top-N review-queue entries. Pure HTML anchors — no JavaScript framework.

## Review queue

Priority formula:

```
priority = max(1, 100 − severity) × (1 + (1 − confidence))
```

- Higher priority → reviewed first.
- Dataset-level findings (no affected annotation) default to `(1 + 1) = 2` on the confidence term.
- Annotation-level with low confidence get a double-boost: lower severity threshold *and* high uncertainty.

Sort tiebreaks: `−priority, severity, filename, (annotation_id is None), annotation_id`.

## Reporting

### `findings_present.py`
Renders the slim, findings-first HTML report. Layout, top to bottom:

1. Header + summary strip (5 severity tiles + review-queue tile with CSV download)
2. **Findings by check** + **Cumulative review burden** (workload plots)
3. **Findings by source** (attribution)
4. **Priority review queue** — ranks link into drill-down anchors
5. **Dataset at a glance** — class composition + confidence-by-class + spatial heatmap
6. **All findings by category** — collapsible `<details>` per category
7. **Drill-down** — per-finding panels with flagged ↔ exemplar thumbnails

Sidecar exports: `review_queue.csv`, `findings.json`, `static/` Bootstrap assets.

### Legacy `present.py`
Untouched. Produces the richer multi-plot report for users who prefer the old layout.

## Extending the tool

### Add a custom check

```python
from lightly_insights.core import Check, Dataset, Finding, Severity, AnnotationKind, register_check

@register_check
class MyCheck(Check):
    check_id = "my_domain_check"
    title = "Something specific to my data"
    category = "annotation"
    supported_kinds = frozenset({AnnotationKind.BOX})

    def run(self, dataset: Dataset) -> list[Finding]:
        findings = []
        for ann in dataset.annotations:
            if self._something_wrong(ann):
                findings.append(Finding(
                    check_id=self.check_id,
                    severity=Severity.HIGH,
                    category=self.category,
                    title="...",
                    detail="...",
                    action="...",
                    affected_annotations=[ann.annotation_id],
                    affected_images=[ann.image_filename],
                ))
        return findings
```

Import your module anywhere before calling `run_all`; registration happens on import.

### Add a custom reporter

`List[Finding]` + `List[ReviewItem]` is the stable interface. Anything that can consume those two lists can render its own view — Slack bot, CI gate, LightlyStudio panel, PDF exporter. Nothing in `core/` is HTML-specific.

### Add a proposal source

```python
from lightly_insights.ml import PROPOSAL_SOURCE_PREFIX
# source naming convention: "proposal:<your-model>"
ann = Annotation(..., source=f"{PROPOSAL_SOURCE_PREFIX}my-model", confidence=0.85)
```

The bundled `missing_label_proposal` check then flags proposals without a matching real annotation.

## What *not* to put here

- **Embedding-based similarity / coreset selection** — LightlyStudio's domain.
- **Interactive labeling** — LightlyStudio / CVAT / Label Studio.
- **Live monitoring / web dashboards** — out of scope for a static-report library.
- **GPU-backed heavy ML** — OK as optional extras (`[ml]`), never required.
