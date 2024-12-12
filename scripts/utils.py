# %%
import pandas as pd
import os
import glob
from enum import Enum

from .constants import features_dir


class DaspsLabeling(Enum):
    HAM = 0
    SAM = 1


class DatasetLabel(Enum):
    CONTROL = 0
    GAD = 1
    SAD = 2


class DatasetEnum(Enum):
    DASPS = 0
    SAD = 1


class LabelingScheme:
    def __init__(
            self, dasps_labeling: DaspsLabeling, *, lo_level_dasps=[0, 1],
            hi_level_dasps=[2, 3],
            lo_level_sad=[0, 1],
            hi_level_sad=[2, 3]) -> None:
        # Make sure low and hi levels do not overlap
        assert len(set(lo_level_dasps) & set(hi_level_dasps)) == 0

        self.dasps_labeling = dasps_labeling
        self.lo_level_dasps = lo_level_dasps
        self.hi_level_dasps = hi_level_dasps
        self.lo_level_sad = lo_level_sad
        self.hi_level_sad = hi_level_sad


def get_extracted_seglens():
    paths = glob.glob(os.path.join(features_dir, 'features_*s.csv'))
    basenames = [os.path.basename(i) for i in paths]

    seglens = [i.strip("features_").strip("s.csv") for i in basenames]
    seglens = [int(i) for i in seglens]

    return seglens


def get_feats_csv_path(seglen: int):
    return os.path.join(features_dir, f'features_{seglen}s.csv')


class BaseDatasetBuilder:
    def _drop_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['dataset', 'ham', 'sam',
                                'stai', 'uniq_subject_id', 'subject'])

    def _get_seglen_df(self, seglen: int) -> pd.DataFrame:
        path = get_feats_csv_path(seglen)

        df = pd.read_csv(path)

        # Remove absolute power features
        # df = df.loc[:, ~df.columns.str.startswith('abs_pow_')]

        return df


# Following labels will be returned:
# 0 - Control
# 1 - GAD
# 2 - SAD
class DatasetBuilder(BaseDatasetBuilder):
    _feat_names: list[str] = []

    def __init__(self, labeling_scheme: LabelingScheme) -> None:
        self._labeling_scheme = labeling_scheme

    def build_dataset_df(
            self, seglen: int) -> pd.DataFrame:
        df = self._get_seglen_df(seglen)
        df = self._label_rows(df)
        df = self._drop_redundant_columns(df)

        self._feat_names = df.columns.tolist()[:-1]

        return df

    def build_dataset_arrs(self, seglen: int):
        df = self.build_dataset_df(seglen)

        # Return three numpy arrays, one for each label
        control = df[df['label'] == DatasetLabel.CONTROL.value].to_numpy()
        gad = df[df['label'] == DatasetLabel.GAD.value].to_numpy()
        sad = df[df['label'] == DatasetLabel.SAD.value].to_numpy()

        # Remove label (last) column
        control = control[:, :-1]
        gad = gad[:, :-1]
        sad = sad[:, :-1]

        return control, gad, sad

    def build_control_dataset_df(self, seglen: int) -> pd.DataFrame:
        df = self._get_seglen_df(seglen)
        df = self._label_rows(df)

        # Include only low anxiety (control) subjects
        df = df[df['label'] == DatasetLabel.CONTROL.value]

        df['label'] = df['dataset'].map(
            {'dasps': DatasetEnum.DASPS.value, 'SAD': DatasetEnum.SAD.value})

        df = self._drop_redundant_columns(df)

        self._feat_names = df.columns.tolist()[:-1]

        return df

    def build_control_dataset_arrs(self, seglen: int):
        df = self.build_control_dataset_df(seglen)

        # Return two numpy arrays, one for each dataset
        dasps = df[df['label'] == DatasetEnum.DASPS.value].to_numpy()
        sad = df[df['label'] == DatasetEnum.SAD.value].to_numpy()

        # Remove label (last) column
        dasps = dasps[:, :-1]
        sad = sad[:, :-1]

        return dasps, sad

    def get_feat_names(self):
        return self._feat_names

    def _label_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Add a column for label
        df['label'] = ""

        for i, row in df.iterrows():
            dataset = row['dataset']

            if dataset == 'dasps':
                label = self._get_dasps_label(row)
            elif dataset == 'SAD':
                label = self._get_sad_label(row)
            else:
                raise ValueError(f'Invalid dataset {dataset}')

            df.at[i, 'label'] = label.name

        return df

    def _get_dasps_label(self, row: pd.Series) -> DatasetLabel:
        if self._labeling_scheme.dasps_labeling.value == DaspsLabeling.HAM.value:
            if row['ham'] in self._labeling_scheme.lo_level_dasps:
                return DatasetLabel.CONTROL
            elif row['ham'] in self._labeling_scheme.hi_level_dasps:
                return DatasetLabel.GAD
            else:
                raise ValueError(f'Invalid HAM severity: {row["ham"]}')

        elif self._labeling_scheme.dasps_labeling.value == DaspsLabeling.SAM.value:
            if row['sam'] in self._labeling_scheme.lo_level_dasps:
                return DatasetLabel.CONTROL
            elif row['sam'] in self._labeling_scheme.hi_level_dasps:
                return DatasetLabel.GAD
            else:
                raise ValueError(f'Invalid SAM severity: {row["sam"]}')

        raise ValueError('Invalid labeling scheme')

    def _get_sad_label(self, row: pd.Series) -> DatasetLabel:
        severity = row['stai']

        if severity in self._labeling_scheme.lo_level_sad:
            return DatasetLabel.CONTROL
        elif severity in self._labeling_scheme.hi_level_sad:
            return DatasetLabel.SAD

        raise ValueError(f'Invalid SAD severity: {severity}')


if __name__ == "__main__":
    # Normal dataset builder
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    df = builder.build_dataset_df(15)

    # Control dataset builder
    dasps, sad = builder.build_control_dataset_arrs(15)

    print(dasps.shape)
    print(sad.shape)

# %%
