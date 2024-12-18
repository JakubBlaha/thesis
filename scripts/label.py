import itertools
import os

from utils import DatasetBuilder, LabelingScheme, DaspsLabeling, get_extracted_seglens
from constants import features_dir


def make_labeled_csv_files():
    extracted_seglens = get_extracted_seglens()

    labeling_schemes = [
        LabelingScheme(DaspsLabeling.HAM),
        LabelingScheme(DaspsLabeling.SAM)]

    for labeling_scheme, seglen in itertools.product(
            labeling_schemes, extracted_seglens):
        builder = DatasetBuilder(labeling_scheme)

        df = builder.build_dataset_df(seglen)

        basename = f"features_{seglen}s_{
            labeling_scheme.dasps_labeling.name}.csv"
        path = os.path.join(features_dir, basename)

        df.to_csv(path, index=False)


if __name__ == "__main__":
    make_labeled_csv_files()
