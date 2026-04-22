"""Command-line entry point for lightly-insights.

Two subcommands:

    lightly-insights report  --images ... --labels ... --format ... --out ./report
    lightly-insights compare ./reportA ./reportB --out diff.md

With no subcommand `report` is assumed so legacy calls keep working.
"""
import argparse
import logging
import sys
from pathlib import Path

from lightly_insights import analyze, compare, present


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


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Folder containing the image files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Folder or file containing the labels (required unless --images-only).",
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


def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "a_folder",
        type=Path,
        help="Folder containing insights.json for dataset A (e.g. the train report).",
    )
    parser.add_argument(
        "b_folder",
        type=Path,
        help="Folder containing insights.json for dataset B (e.g. the val report).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown file. Defaults to <a_folder>/compare_<b>.md.",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="Label for dataset A in the diff (defaults to the folder name).",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="Label for dataset B in the diff (defaults to the folder name).",
    )


def _run_report(args: argparse.Namespace) -> int:
    if not args.images_only and (args.labels is None or args.format is None):
        print(
            "error: --labels and --format are required unless --images-only is set.",
            file=sys.stderr,
        )
        return 2

    image_analysis = analyze.analyze_images(
        image_folder=args.images,
        recursive=args.recursive,
        check_quality=not args.no_quality_check,
        find_near_duplicates=args.find_near_duplicates,
    )

    if args.images_only:
        od_analysis = analyze.ObjectDetectionAnalysis(
            num_images=image_analysis.num_images,
            num_images_zero_objects=image_analysis.num_images,
            filename_set=image_analysis.filename_set,
            total=analyze.ClassAnalysis.create_empty(id=-1, name="[All classes]"),
            classes={},
        )
    else:
        input_cls = _import_format(args.format)
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


def _run_compare(args: argparse.Namespace) -> int:
    out = args.out or (args.a_folder / f"compare_{args.b_folder.name}.md")
    path = compare.write_comparison(
        a_folder=args.a_folder,
        b_folder=args.b_folder,
        output_file=out,
        a_label=args.label_a,
        b_label=args.label_b,
    )
    print(f"Comparison written to {path.resolve()}")
    return 0


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lightly-insights",
        description=(
            "Generate a dataset insights HTML report and compare runs across "
            "datasets or iterations."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    report_p = subparsers.add_parser(
        "report", help="Generate an insights report (default command)."
    )
    _add_report_args(report_p)

    compare_p = subparsers.add_parser(
        "compare", help="Diff two previously-generated insights reports."
    )
    _add_compare_args(compare_p)

    # Backwards compatibility: `lightly-insights --images ... --labels ...`
    # with no subcommand should act like `report`.
    known_subs = {"report", "compare"}
    av = list(argv) if argv is not None else sys.argv[1:]
    if not av or av[0] not in known_subs:
        av = ["report", *av]

    args = parser.parse_args(av)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "report":
        return _run_report(args)
    if args.command == "compare":
        return _run_compare(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
