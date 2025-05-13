import argparse
import logging

from convert import convert_dasps_to_fif, convert_sad_to_fif
from segment import segment, validate_seglen
from autoreject_runner import run_autoreject
from extract_features import extract_features_from_all_segments
from training import train_models
from deep import run_deep_learning
from ensemble import train_model as run_ensemble
from metrics import process_latest_result_file
from metrics import process_results_file

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
                logger.warning(
                    f"Invalid segment length: {seglen_int}. Valid values are {valid_seglens}")
        except ValueError:
            logger.warning(
                f"Invalid segment length: {seglen}. Must be an integer.")

    if not seglens:
        raise ValueError(
            f"No valid segment lengths provided. Valid values are {valid_seglens}")

    return seglens


def parse_classifiers(classifiers_str):
    """Parse comma-separated classifiers string into a list."""
    if not classifiers_str:
        return []
    return [clf.strip() for clf in classifiers_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="EEG data processing and machine learning pipeline for sleep stage classification."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
        help="Select a command to execute"
    )

    command_descriptions = {
        "convert": "Convert DASPS and SAD datasets to FIF format",
        "segment": "Segment the EEG data into fixed-length segments",
        "autoreject": "Apply autoreject to clean EEG data",
        "extract": "Extract features from segmented EEG data",
        "train": "Train machine learning models on extracted features",
        "deep": "Run deep learning models (LSTM or CNN) on EEG data",
        "ensemble": "Create ensemble models using voting or stacking strategies",
        "metrics": "Calculate and visualize metrics for model evaluation"
    }

    available_commands = list(command_descriptions.keys())

    cmd_parsers = {}

    for command, description in command_descriptions.items():
        cmd_parsers[command] = subparsers.add_parser(
            command,
            help=description,
            description=description
        )

    for parser_name in ["segment", "extract"]:
        cmd_parsers[parser_name].add_argument(
            "--seglen", type=int,
            help="Segment length in seconds (valid values: 1, 2, 3, 5, 10, 15, 30)",
            required=True)

    cmd_parsers["train"].add_argument(
        "--labeling-scheme",
        choices=["ham", "sam"],
        help="DASPS labeling scheme to use (ham or sam)",
        required=True,
    )

    cmd_parsers["train"].add_argument(
        "--domains",
        default="rel_pow,conn,ai,time,abs_pow",
        help="Comma-separated list of domains to use (default: rel_pow,conn,ai,time,abs_pow)",
    )

    cmd_parsers["train"].add_argument(
        "--mode",
        choices=["both", "dasps", "sad"],
        default="both",
        help="Dataset mode for training (both: use both datasets, dasps: use only DASPS, sad: use only SAD)",
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
        help="Cross-validation strategy (logo: leave-one-group-out, skf: stratified k-fold)",
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

    cmd_parsers["train"].add_argument(
        "--plot-roc",
        action="store_true",
        help="Plot ROC curves for the trained models",
    )

    cmd_parsers["deep"].add_argument(
        "--seglen", type=int,
        help="Segment length in seconds (valid values: 1, 2, 3, 5, 10, 15, 30)",
        required=True)

    cmd_parsers["deep"].add_argument(
        "--classif", choices=["lstm", "cnn"],
        help="Deep learning classifier type (lstm: Long Short-Term Memory, cnn: Convolutional Neural Network)",
        required=True)

    cmd_parsers["ensemble"].add_argument(
        "--strategy",
        required=True,
        choices=["voting", "stacking"],
        help="Ensemble strategy: voting (majority vote) or stacking (meta-classifier)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--seglen",
        type=int,
        default=15,
        help="Segment length in seconds (valid values: 1, 2, 3, 5, 10, 15, 30; default: 15)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--mode",
        choices=["both", "dasps", "sad"],
        default="both",
        help="Dataset mode for training (both: use both datasets, dasps: use only DASPS, sad: use only SAD)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--domains",
        default="rel_pow,conn,ai,time,abs_pow",
        help="Comma-separated list of domains to use (default: rel_pow,conn,ai,time,abs_pow)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--no-oversample",
        action="store_true",
        help="Disable oversampling (enabled by default)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--final-classifier",
        choices=["logistic", "rf", "mlp", "gb"],
        default="logistic",
        help="Final classifier for stacking (logistic: Logistic Regression, rf: Random Forest, mlp: Multi-layer Perceptron, gb: Gradient Boosting; default: logistic)",
    )

    cmd_parsers["ensemble"].add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    cmd_parsers["metrics"].add_argument(
        "--file",
        help="Path to the results CSV file (optional, uses latest file if not provided)"
    )

    cmd_parsers["metrics"].add_argument(
        "--title",
        help="Title for the confusion matrix plot (optional)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

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

    if args.command == "train":
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
        plot_roc = args.plot_roc

        if not classifiers:
            logger.error("No valid classifiers provided")
            return

        logger.info(f"Training with classifiers: {classifiers}, seglens: {seglens}, mode: {mode}, domains: {domains}, "
                    f"labeling scheme: {dasps_labeling_scheme}, oversample: {oversample}, cv: {cv}, plot_roc: {plot_roc}")

        train_models(seglens=seglens, mode=mode, domains=domains,
                     dasps_labeling_scheme=dasps_labeling_scheme,
                     oversample=oversample, cv=cv, classifiers=classifiers,
                     plot_roc=plot_roc)
        logger.info("Training completed.")

    if args.command == "deep":
        seglen = args.seglen
        classif = args.classif

        validate_seglen(seglen)

        logger.info(
            f"Running deep learning with seglen={seglen}, classifier={classif}")
        run_deep_learning(seglen=seglen, model_type_param=classif)
        logger.info("Deep learning completed.")

    if args.command == "ensemble":
        strategy = args.strategy
        seglen = args.seglen
        mode = args.mode
        domains = parse_domains(args.domains)
        oversample = not args.no_oversample
        final_classifier = args.final_classifier
        seed = args.seed

        logger.info(f"Running ensemble with strategy={strategy}, seglen={seglen}, "
                    f"mode={mode}, domains={domains}, oversample={oversample}, "
                    f"final_classifier={final_classifier}, seed={seed}")

        run_ensemble(
            seglen=seglen,
            mode=mode,
            domains=domains,
            strategy=strategy,
            final_classifier=final_classifier,
            oversample=oversample,
            seed=seed
        )

        logger.info("Ensemble training completed.")

    if args.command == "metrics":
        if args.file:
            logger.info(f"Calculating metrics for file: {args.file}")
            process_results_file(args.file, args.title)
        else:
            logger.info("Calculating metrics for latest results file")
            process_latest_result_file()

        logger.info("Metrics calculation completed.")


if __name__ == '__main__':
    main()
