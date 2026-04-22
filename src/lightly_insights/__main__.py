"""Command-line entry point for lightly-insights.

Usage:
    python -m lightly_insights --images ./imgs --labels ./ann --format coco --out ./report
"""
import argparse
import logging
import sys
from pathlib import Path

from lightly_insights import analyze, present


_LABEL_FORMATS = {
    "coco": "labelformat.formats.COCOObjectDetectionInput",
    "pascalvoc": "labelformat.formats.PascalVOCObjectDetectionInput",
    "yolov5": "labelformat.formats.YOLOv5ObjectDetectionInput",
    "yolov8": "labelformat.formats.YOLOv8ObjectDetectionInput",
}


def _import_format(name: str) -> type:
    """Import a labelformat input class lazily so users only pay for what they use."""
    dotted = _LABEL_FORMATS[name]
    module_path, cls_name = dotted.rsplit(".", 1)
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lightly-insights",
        description=(
            "Generate a dataset insights HTML report from an image folder and "
            "object-detection labels."
        ),
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Folder containing the image files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help=(
            "Folder or file containing the labels. Required unless "
            "--images-only is passed."
        ),
    )
    parser.add_argument(
        "--format",
        choices=sorted(_LABEL_FORMATS.keys()),
        help="Label format (required when --labels is given).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output folder for the HTML report.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan the image folder recursively (includes subdirectories).",
    )
    parser.add_argument(
        "--find-near-duplicates",
        action="store_true",
        help=(
            "Scan for near-duplicate images via perceptual hash. "
            "Requires the 'imagehash' package."
        ),
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip the image-quality scan (all-black, blur, aspect outliers).",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only analyze images; skip object-detection report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.images_only and (args.labels is None or args.format is None):
        parser.error("--labels and --format are required unless --images-only is set.")

    image_analysis = analyze.analyze_images(
        image_folder=args.images,
        recursive=args.recursive,
        check_quality=not args.no_quality_check,
        find_near_duplicates=args.find_near_duplicates,
    )

    if args.images_only:
        # Build a minimal OD analysis so the template still renders.
        od_analysis = analyze.ObjectDetectionAnalysis(
            num_images=image_analysis.num_images,
            num_images_zero_objects=image_analysis.num_images,
            filename_set=image_analysis.filename_set,
            total=analyze.ClassAnalysis.create_empty(id=-1, name="[All classes]"),
            classes={},
        )
    else:
        input_cls = _import_format(args.format)
        # Labelformat input classes take different kwargs depending on format.
        # The common one is input_folder / input_file; try the folder form
        # first, then fall back to file.
        try:
            label_input = input_cls(input_folder=args.labels)
        except TypeError:
            label_input = input_cls(input_file=args.labels)
        od_analysis = analyze.analyze_object_detections(label_input=label_input)

    present.create_html_report(
        output_folder=args.out,
        image_analysis=image_analysis,
        od_analysis=od_analysis,
    )
    print(f"Report written to {args.out.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
