import argparse
import logging

from convert import convert_dasps_to_fif, convert_sad_to_fif
from segment import segment, validate_seglen
from autoreject_runner import run_autoreject
from extract_features import extract_features_from_all_segments
from label import make_labeled_csv_files
from training import train_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_domains(domains_str):
    """Parse comma-separated domains string into a list."""
    if not domains_str:
        return ["rel_pow", "conn", "ai", "time", "abs_pow"]
    return [domain.strip() for domain in domains_str.split(',')]

def parse_seglens(seglens_str):
    """Parse comma-separated segment lengths string into a list of integers."""
    if not seglens_str:
        return []
    valid_seglens = [1, 2, 3, 5, 10, 15, 30]
    seglens = []
    for seglen in seglens_str.split(','):
        try:
            seglen_int = int(seglen.strip())
            if seglen_int in valid_seglens:
                seglens.append(seglen_int)
            else:
                logger.warning(f"Invalid segment length: {seglen_int}. Valid values are {valid_seglens}")
        except ValueError:
            logger.warning(f"Invalid segment length: {seglen}. Must be an integer.")
    
    if not seglens:
        raise ValueError(f"No valid segment lengths provided. Valid values are {valid_seglens}")
    
    return seglens

def parse_classifiers(classifiers_str):
    """Parse comma-separated classifiers string into a list."""
    if not classifiers_str:
        return []
    return [clf.strip() for clf in classifiers_str.split(',')]

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    available_commands = ["convert", "segment",
                          "autoreject", "extract", "label", "train"]

    cmd_parsers = {}

    for command in available_commands:
        cmd_parsers[command] = subparsers.add_parser(command)

    for parser_name in ["segment", "extract"]:
        cmd_parsers[parser_name].add_argument(
            "--seglen",
            type=int,
            help="Segment length in seconds",
            required=True
        )
        
    # Add arguments for train command
    cmd_parsers["train"].add_argument(
        "--labeling-scheme",
        choices=["ham", "sam"],
        help="DASPS labeling scheme to use (ham or sam)",
        required=True,
    )
    
    cmd_parsers["train"].add_argument(
        "--domains",
        default="rel_pow,conn,ai,time,abs_pow",
        help="Comma-separated list of domains to use (default: all domains)",
    )
    
    cmd_parsers["train"].add_argument(
        "--mode",
        choices=["both", "dasps", "sad"],
        default="both",
        help="Dataset mode for training (default: both)",
    )
    
    cmd_parsers["train"].add_argument(
        "--no-oversample",
        action="store_true",
        help="Disable oversampling (enabled by default)",
    )
    
    cmd_parsers["train"].add_argument(
        "--cv",
        choices=["logo", "skf"],
        required=True,
        help="Cross-validation strategy",
    )
    
    cmd_parsers["train"].add_argument(
        "--seglens",
        required=True,
        help="Comma-separated list of segment lengths to use (e.g., '1,2,5'). Valid values: 1, 2, 3, 5, 10, 15, 30",
    )
    
    cmd_parsers["train"].add_argument(
        "--classifiers",
        required=True,
        help="Comma-separated list of classifiers to use (e.g., 'svm-rbf,rf,knn')",
    )

    args = parser.parse_args()

    if not args.command or args.command not in available_commands:
        parser.print_help()

    if args.command == "convert":
        convert_dasps_to_fif()
        convert_sad_to_fif()

        logger.info("Converted DASPS and SAD datasets to FIF.")

    if args.command == "segment":
        seglen = args.seglen

        validate_seglen(seglen)
        segment(seglen)

        logger.info("Segmentation completed.")

    if args.command == "autoreject":
        run_autoreject()

        logger.info("Autoreject completed.")

    if args.command == "extract":
        extract_features_from_all_segments(seglen=args.seglen)

        logger.info("Feature extraction completed.")

    if args.command == "label":
        make_labeled_csv_files()

        logger.info("Labeling completed.")

    if args.command == "train":
        # Parse segment lengths from command line argument
        try:
            seglens = parse_seglens(args.seglens)
        except ValueError as e:
            logger.error(str(e))
            return
            
        domains = parse_domains(args.domains)
        mode = args.mode
        dasps_labeling_scheme = args.labeling_scheme
        oversample = not args.no_oversample
        cv = args.cv
        classifiers = parse_classifiers(args.classifiers)
        
        if not classifiers:
            logger.error("No valid classifiers provided")
            return

        logger.info(f"Training with classifiers: {classifiers}, seglens: {seglens}, mode: {mode}, domains: {domains}, "
                    f"labeling scheme: {dasps_labeling_scheme}, oversample: {oversample}, cv: {cv}")

        train_models(seglens=seglens, mode=mode, domains=domains,
                     dasps_labeling_scheme=dasps_labeling_scheme,
                     oversample=oversample, cv=cv, classifiers=classifiers)
        logger.info("Training completed.")


if __name__ == '__main__':
    main()
