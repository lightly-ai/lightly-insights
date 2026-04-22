"""Tests for drill-down panels + source attribution."""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from lightly_insights.core import (
    Annotation,
    AnnotationKind,
    Category,
    Dataset,
    Finding,
    Image,
    ReviewItem,
    Severity,
    build_review_queue,
)
from lightly_insights.core.attribution import (
    compute_attribution,
    render_attribution_plot,
)
from lightly_insights.core.drilldown import build_drilldowns
from lightly_insights.core.geometry import Box


def _paint(folder: Path, name: str, size=(200, 200), color=(120, 120, 120)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    PILImage.new("RGB", size, color=color).save(path)
    return path


def _ann(idx, filename, cid, name, path, box, source=None):
    return Annotation(
        annotation_id=idx, image_filename=filename, class_id=cid,
        class_name=name, kind=AnnotationKind.BOX, geometry=box, source=source,
    )


def test_build_drilldowns_renders_flagged_and_exemplar_thumbnails(tmp_path: Path) -> None:
    # Two annotations of class car: one flagged, one exemplar.
    img_dir = tmp_path / "imgs"
    flagged_path = _paint(img_dir, "flagged.jpg", color=(200, 50, 50))
    exemplar_path = _paint(img_dir, "ok.jpg", color=(80, 150, 80))

    flagged = Annotation(
        annotation_id=0, image_filename="flagged.jpg", class_id=0,
        class_name="car", kind=AnnotationKind.BOX, geometry=Box(10, 10, 80, 80),
    )
    exemplar = Annotation(
        annotation_id=1, image_filename="ok.jpg", class_id=0,
        class_name="car", kind=AnnotationKind.BOX, geometry=Box(20, 20, 70, 70),
    )
    ds = Dataset(
        images=[
            Image(filename="flagged.jpg", width=200, height=200, path=flagged_path),
            Image(filename="ok.jpg", width=200, height=200, path=exemplar_path),
        ],
        annotations=[flagged, exemplar],
        categories=[Category(id=0, name="car")],
    )

    findings = [
        Finding(
            check_id="shape_outlier",
            severity=Severity.MEDIUM,
            category="annotation",
            title="Outlier",
            detail="details",
            action="fix it",
            affected_annotations=[0],
            affected_images=["flagged.jpg"],
        )
    ]
    queue = build_review_queue(findings, ds)
    panels = build_drilldowns(
        output_folder=tmp_path,
        dataset=ds,
        findings=findings,
        review_queue=queue,
    )
    assert len(panels) == 1
    panel = panels[0]
    assert panel.anchor == "finding-1"
    assert (tmp_path / panel.flagged_src).exists()
    # Exemplar should be rendered too.
    assert panel.exemplar_src != ""
    assert (tmp_path / panel.exemplar_src).exists()


def test_build_drilldowns_skips_when_image_missing(tmp_path: Path) -> None:
    ds = Dataset(
        images=[Image(filename="missing.jpg", width=100, height=100, path=tmp_path / "missing.jpg")],
        annotations=[
            Annotation(
                annotation_id=0, image_filename="missing.jpg", class_id=0,
                class_name="x", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            )
        ],
        categories=[Category(id=0, name="x")],
    )
    findings = [
        Finding(
            check_id="c", severity=Severity.MEDIUM, category="annotation",
            title="t", detail="d", action="a",
            affected_annotations=[0], affected_images=["missing.jpg"],
        )
    ]
    queue = build_review_queue(findings, ds)
    panels = build_drilldowns(
        output_folder=tmp_path, dataset=ds, findings=findings, review_queue=queue,
    )
    assert panels == []


def test_compute_attribution_ranks_worst_source_first() -> None:
    # yolo: 10 annotations, 5 findings. sam: 10 annotations, 1 finding.
    anns = [
        Annotation(
            annotation_id=i, image_filename="a.jpg", class_id=0,
            class_name="car", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            source="yolo",
        ) for i in range(10)
    ] + [
        Annotation(
            annotation_id=i + 10, image_filename="a.jpg", class_id=0,
            class_name="car", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            source="sam",
        ) for i in range(10)
    ]
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=anns,
        categories=[Category(id=0, name="car")],
    )
    findings = [
        Finding(
            check_id="check_a", severity=Severity.MEDIUM, category="annotation",
            title="t", detail="d", action="a", affected_annotations=[i],
        ) for i in range(5)  # all touch yolo
    ] + [
        Finding(
            check_id="check_b", severity=Severity.MEDIUM, category="annotation",
            title="t", detail="d", action="a", affected_annotations=[15],
        )  # one touches sam
    ]
    items = compute_attribution(ds, findings)
    assert len(items) == 2
    assert items[0].source == "yolo"
    assert items[0].num_findings == 5
    assert items[0].findings_per_100 == 50.0
    assert items[1].source == "sam"
    assert items[1].num_findings == 1


def test_compute_attribution_empty_when_no_sources() -> None:
    ds = Dataset(
        images=[Image(filename="a.jpg", width=100, height=100)],
        annotations=[
            Annotation(
                annotation_id=0, image_filename="a.jpg", class_id=0,
                class_name="x", kind=AnnotationKind.BOX, geometry=Box(0, 0, 10, 10),
            )
        ],
        categories=[Category(id=0, name="x")],
    )
    assert compute_attribution(ds, []) == []


def test_render_attribution_plot(tmp_path: Path) -> None:
    from lightly_insights.core.attribution import AttributionItem
    items = [
        AttributionItem(source="yolo", num_annotations=100, num_findings=20,
                        findings_per_100=20.0, top_check_ids=[("x", 10)]),
        AttributionItem(source="sam", num_annotations=50, num_findings=2,
                        findings_per_100=4.0, top_check_ids=[("y", 2)]),
    ]
    path = render_attribution_plot(tmp_path, items)
    assert path == "attribution.png"
    assert (tmp_path / "attribution.png").exists()
