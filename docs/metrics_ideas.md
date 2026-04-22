# Dataset-quality metrics worth adding

Curated list of high-value metrics not yet implemented, ranked by expected leverage. Shortlist at the bottom.

## Autolabel-centric (highest value in the current direction)

### 1. Expected calibration error (ECE) from a reviewed sample
If a subset of autolabels has been human-reviewed, bucket predictions by confidence (e.g. 10-bin histogram of 0.0-1.0) and compare each bin's accuracy to its mean confidence. A well-calibrated autolabeler shows `|accuracy - confidence|` near zero in every bin. High ECE means the autolabeler's confidence is lying — reviewers can't trust thresholds.

**Needs:** A boolean "reviewed & correct" signal on some annotations. Store via new `Annotation.reviewed: Optional[bool]` field.

### 2. TTA robustness: consistency under augmentation
Run the autolabeler on `(image, flip(image), rotate(image), jitter(image))` and measure IoU of corresponding annotations. Low IoU = the autolabeler is brittle; those predictions should be treated with extra scrutiny.

**Needs:** `[ml]` extra. Can cache per-image robustness scores on the Annotation itself.

### 3. Multi-seed / multi-threshold stability
Run the same autolabeler at confidence thresholds {0.3, 0.5, 0.7} or with different seeds. Predictions that appear under all settings are high-confidence. Those that appear only at the loosest threshold are noise. Emits a stability score per annotation, feeds into the review queue sort.

### 4. Proposal ↔ prediction gap via a second model
Run a *different* autolabeler (e.g., SAM for segmentation vs Grounding-DINO for boxes) and check whether the two agree. Multi-source disagreement exists but uses pre-existing annotations. This flavor runs a proposer fresh for comparison.

## Geometry & shape (under-explored)

### 5. Bounding-box "hugging" quality
Edge detector on the image; check what fraction of each box's border lies within 2 px of a high-gradient pixel. Loose/sloppy boxes → low score. This is domain-agnostic and cheap (one Sobel per image).

**Visualization:** heatmap per class showing where the "sloppiness" concentrates.

### 6. Per-class shape prototype coherence
Project each annotation into a feature vector (area, aspect ratio, tight-box normalized center, maybe HOG features), compute the class centroid, then the intra-class spread. Classes with high spread have "examples of everything" — good or a sign of ambiguous labeling.

### 7. Class centroid confusion
In the feature space above, compute pairwise centroid distances between classes. Small distances = classes that overlap visually (e.g. "car" vs "SUV"). Predicts which class pairs will confuse a detector.

### 8. Tiny-hole detection in masks
Flood-fill from the image border on the inverted mask; isolate interior holes. Holes < 16 px are almost always artifacts of rasterization or labeling slips. Cheap, mask-only.

## Distribution / bias (high-impact, not yet shipped)

### 9. Fine-grained cross-field correlations
- Per-camera class distribution drift (when EXIF is available).
- Per-time-of-day class distribution drift.
- Per-annotator class distribution (bias indicator).
- Per-filename-prefix distribution drift (catches "train/" vs "val/" that differ in ways beyond class counts).

Implementation is small: one `χ²` call per axis, thresholds identical to `split_purity`.

### 10. Scene-complexity estimate per image
Proxies: number of annotations, variance of color, edge density. Feeds into a "hard images" list for targeted review. Cheap, purely pixel-level.

### 11. Long-tail shape analysis
Beyond the single "imbalance score" (Gini, entropy), fit a power-law to the class-count distribution: Zipfian α. α ≥ 2 means "classic long tail"; α < 1 means "mostly uniform." Different α → different mitigation strategies (oversampling vs. class weighting vs. hierarchical classification).

### 12. Image-embedding diversity (optional, heavy)
Hash or small-CNN features per image. Report intra-dataset diversity (mean pairwise distance) and per-class diversity. Low = dataset is narrow, model won't generalize. **Conflicts with LightlyStudio's domain — probably skip here and let Studio own this.**

## Reviewer process (non-statistical)

### 13. Cost-aware review budget
User specifies: "I have 2 hours, annotations cost 30 s each." Tool returns the top-N findings that fit in that budget, optimizing expected catch-rate per minute. Just a knapsack over the existing priority-sorted queue.

### 14. "Review to green" simulator
If the reviewer accepts / rejects the top 10 findings as you'd predict, what's the resulting health score? Simulates the impact of spending N minutes on review before the user commits. Drives realistic time estimates.

### 15. Labeling-tool export
Labelbox / CVAT / V7 / Label Studio all have JSON formats for "here are tasks to review." Emit `review_queue.csv` in those formats directly, so no glue code is needed between the insights run and the review session.

## Temporal / sequential (only when the data supports it)

### 16. Inter-frame label continuity
When filenames imply frame order (`seq_0001.jpg`, `seq_0002.jpg`, ...), associate boxes across frames via IoU + centroid velocity. Flag frames where an object disappears and reappears, or where class changes without corresponding motion. Catches frame-skip and reset bugs in video labeling.

### 17. Drift across autolabeler versions
When the dataset carries `source="yolo-v8"` and `source="yolo-v11"` on the same images, measure per-class count deltas, per-annotation class changes, new/dropped annotations. Tells you the cost of upgrading the labeler.

## Would NOT add

- **Custom ML models trained on the dataset itself** — scope creep; out of a static-analysis library's lane.
- **Full embeddings pipeline** — LightlyStudio territory.
- **Real-time monitoring** — static-report model doesn't fit.
- **Generic "image aesthetics" scoring** — not related to training utility.

---

## Shortlist: next five I'd build

If continuing on this tool, in order of expected ROI:

1. **#1 Expected calibration error** — lands after a `reviewed` field is added. Single most useful number for an autolabel-driven team.
2. **#5 Bounding-box hugging quality** — domain-agnostic, cheap, finds real labeling sloppiness humans can't eyeball at scale.
3. **#9 Cross-field correlations** — tiny implementation, surfaces biases no single-axis check catches.
4. **#13 Cost-aware review budget** — shifts the tool from "here are 500 findings" to "review these 40 in your 2-hour window."
5. **#16 Inter-frame continuity** — targeted at video-labeled datasets; nothing else in the tool covers this angle.

Items 2, 6, 8 are the best pure-segmentation additions. Items 1, 13 are the best autolabel-workflow additions.
