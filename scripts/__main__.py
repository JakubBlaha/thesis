import argparse
from .convert import convert_dasps_to_fif, convert_sad_to_fif
from .segment import segment, validate_seglen
from .autoreject_runner import run_autoreject
from .extract_features import extract_features_from_all_segments


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seglen",
        type=int,
        help="Segment length in seconds",
        required=True
    )

    args = parser.parse_args()

    seglen = args.seglen

    validate_seglen(seglen)

    convert_dasps_to_fif()
    convert_sad_to_fif()

    segment(seglen)

    run_autoreject()

    extract_features_from_all_segments()


if __name__ == '__main__':
    main()
