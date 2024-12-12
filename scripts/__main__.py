import argparse
import logging

from .convert import convert_dasps_to_fif, convert_sad_to_fif
from .segment import segment, validate_seglen
from .autoreject_runner import run_autoreject
from .extract_features import extract_features_from_all_segments
from .label import make_labeled_csv_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    available_commands = ["convert", "segment",
                          "autoreject", "extract", "label", "all"]

    cmd_parsers = {}

    for command in available_commands:
        cmd_parsers[command] = subparsers.add_parser(command)

    for parser_name in ["segment", "all"]:
        cmd_parsers[parser_name].add_argument(
            "--seglen",
            type=int,
            help="Segment length in seconds",
            required=True
        )

    args = parser.parse_args()

    _run_all = args.command == "all"

    if not args.command or args.command not in available_commands:
        parser.print_help()

    if args.command == "convert" or _run_all:
        convert_dasps_to_fif()
        convert_sad_to_fif()

        logger.info("Converted DASPS and SAD datasets to FIF.")

    if args.command == "segment" or _run_all:
        seglen = args.seglen

        validate_seglen(seglen)
        segment(seglen)

        logger.info("Segmentation completed.")

    if args.command == "autoreject" or _run_all:
        run_autoreject()

        logger.info("Autoreject completed.")

    if args.command == "extract" or _run_all:
        extract_features_from_all_segments()

        logger.info("Feature extraction completed.")

    if args.command == "label" or _run_all:
        make_labeled_csv_files()

        logger.info("Labeling completed.")


if __name__ == '__main__':
    main()
