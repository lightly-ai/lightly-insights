"""Run a frozen YOLO model and emit proposal annotations.

Uses ultralytics. Kept behind the ``[ml]`` extra because torch is heavy.

Proposals are tagged ``source="proposal:<model-name>"`` so the bundled
``missing_label_proposal`` check can distinguish them from real labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from lightly_insights.core.dataset import Annotation, AnnotationKind
from lightly_insights.core.geometry import Box

# Source names starting with this prefix are treated as proposals by the
# missing-label check. Callers can tag their own proposals similarly.
PROPOSAL_SOURCE_PREFIX = "proposal:"


def is_proposal_source(source: Optional[str]) -> bool:
    return source is not None and source.startswith(PROPOSAL_SOURCE_PREFIX)


def propose_with_yolo(
    image_paths: Iterable[Path],
    model: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    starting_annotation_id: int = 10_000_000,
    category_name_to_id: Optional[Dict[str, int]] = None,
) -> List[Annotation]:
    """Run a pretrained YOLO on each image; return proposal annotations.

    - ``model``: a checkpoint name ultralytics can load (``"yolov8n.pt"``,
      ``"yolov8s.pt"``, ...) or a local path.
    - ``confidence_threshold``: predictions below this are dropped.
    - ``starting_annotation_id``: proposal ids start here so they don't
      collide with your dataset's real annotation ids. Defaults to 10M
      which is a safe zone for most datasets.
    - ``category_name_to_id``: maps YOLO's label strings
      (e.g. ``"person"``) to your dataset's class ids. Names not in the
      mapping get a negative class id — you can still run
      missing-label detection on them.

    Raises ``ImportError`` if ``ultralytics`` isn't installed (the
    ``[ml]`` extra).
    """
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "lightly_insights[ml] is required for YOLO proposals. "
            "Install with: pip install 'lightly-insights[ml]'"
        ) from exc

    yolo = YOLO(model)
    source_tag = f"{PROPOSAL_SOURCE_PREFIX}{Path(model).stem}"
    category_name_to_id = category_name_to_id or {}
    annotations: List[Annotation] = []
    next_id = starting_annotation_id

    for image_path in image_paths:
        results = yolo.predict(
            source=str(image_path), conf=confidence_threshold, verbose=False
        )
        for result in results:
            if result.boxes is None:
                continue
            names = result.names  # class index -> label string
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = names.get(cls_idx, str(cls_idx))
                class_id = category_name_to_id.get(label, -1)
                annotations.append(
                    Annotation(
                        annotation_id=next_id,
                        image_filename=image_path.name,
                        class_id=class_id,
                        class_name=label,
                        kind=AnnotationKind.BOX,
                        geometry=Box(
                            xmin=xyxy[0], ymin=xyxy[1],
                            xmax=xyxy[2], ymax=xyxy[3],
                        ),
                        confidence=conf,
                        source=source_tag,
                    )
                )
                next_id += 1
    return annotations
