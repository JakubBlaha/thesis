# %%
import pandas as pd
import os
import glob
from enum import Enum
from scipy.stats import f_oneway

from constants import features_dir


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
                                'stai', 'subject'])

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
    _feat_domain_prefix = ['time', 'abs_pow', 'rel_pow', 'conn', 'ai']

    def __init__(self, labeling_scheme: LabelingScheme) -> None:
        self._labeling_scheme = labeling_scheme

    def build_dataset_df(self, seglen: int, mode="both",
                         domains:
                         list[str] = ["time", "rel_pow", "conn", "ai"],
                         p_val_thresh=0.05) -> pd.DataFrame:

        df = self._get_seglen_df(seglen)
        df = self._label_rows(df)
        df = self._keep_mode_rows(df, mode)
        df = self._drop_redundant_columns(df)
        df = self._keep_feat_cols(df, domains)
        df = self._keep_significant_cols(df, p_val_thresh)

        self._feat_names = df.columns.tolist()[:-2]

        return df

    def build_dataset_arrs(self, seglen: int):
        df = self.build_dataset_df(seglen)

        control_df = df[df['label'] == DatasetLabel.CONTROL.value]
        gad_df = df[df['label'] == DatasetLabel.GAD.value]
        sad_df = df[df['label'] == DatasetLabel.SAD.value]

        control_df = control_df.drop(columns=['label', 'uniq_subject_id'])
        gad_df = gad_df.drop(columns=['label', 'uniq_subject_id'])
        sad_df = sad_df.drop(columns=['label', 'uniq_subject_id'])

        return control_df.to_numpy(), gad_df.to_numpy(), sad_df.to_numpy()

    def build_control_dataset_df(self, seglen: int) -> pd.DataFrame:
        df = self._get_seglen_df(seglen)
        df = self._label_rows(df)

        # Include only low anxiety (control) subjects
        df = df[df['label'] == DatasetLabel.CONTROL.value]

        df['label'] = df['dataset'].map(
            {'dasps': DatasetEnum.DASPS.value, 'SAD': DatasetEnum.SAD.value})

        df = self._drop_redundant_columns(df)

        self._feat_names = df.columns.tolist()[:-2]

        return df

    def build_control_dataset_arrs(self, seglen: int):
        df = self.build_control_dataset_df(seglen)

        # label is a DatasetEnum value
        dasps_df = df[df['label'] == DatasetEnum.DASPS.value]
        sad_df = df[df['label'] == DatasetEnum.SAD.value]

        # Remove label and uniq_subject_id
        dasps_df = dasps_df.drop(columns=['label', 'uniq_subject_id'])
        sad_df = sad_df.drop(columns=['label', 'uniq_subject_id'])

        return dasps_df.to_numpy(), sad_df.to_numpy()

    def get_feat_names(self):
        return self._feat_names

    def _keep_feat_cols(
            self, df: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
        assert all([d in self._feat_domain_prefix
                    for d in domains]), "Invalid domain"

        print("Keeping domains:", domains)

        if len(domains) == 0:
            domains += ["time", "rel_pow", "conn", "ai"]

        domains += ['dataset', 'label', "uniq_subject_id"]

        # Keep only the selected domains
        cols = [col for col in df.columns if any(
            [col.startswith(d) for d in domains])]

        return df[cols]

    def _get_feat_col_names(self, df: pd.DataFrame) -> list[str]:
        return [col for col in df.columns if any(
            [col.startswith(d) for d in self._feat_domain_prefix])]

    def _keep_significant_cols(
            self, df: pd.DataFrame, p_val_thresh: float) -> pd.DataFrame:
        control = df[df['label'] == DatasetLabel.CONTROL.value].copy()
        gad = df[df['label'] == DatasetLabel.GAD.value].copy()
        sad = df[df['label'] == DatasetLabel.SAD.value].copy()

        for group in [control, gad, sad]:
            group.drop(columns=['label', 'uniq_subject_id'],
                       inplace=True)

        p_vals = {}

        nonempty_groups = [i for i in [control, gad, sad] if not i.empty]

        for col in control.columns:
            f_val, p_val = f_oneway(*[group[col] for group in nonempty_groups])
            p_vals[col] = p_val

        # print(p_vals)

        for col in control.columns:
            is_feature = any([col.startswith(d)
                             for d in self._feat_domain_prefix])

            if p_vals[col] > p_val_thresh and is_feature:
                df = df.drop(columns=[col])

        return df

    def _keep_mode_rows(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        _valid_modes = ["both", "dasps", "sad"]

        assert mode in _valid_modes, "mode must be one of " + str(
            _valid_modes)

        if mode == "both":
            return df
        elif mode == "dasps":
            return df[df['dataset'] == 'dasps']
        elif mode == "sad":
            return df[df['dataset'] == 'SAD']

        raise ValueError(f'Invalid mode: {mode}')

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

            df.at[i, 'label'] = label.value

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

    df = builder.build_dataset_df(10)

    # Control dataset builder
    # dasps, sad = builder.build_control_dataset_arrs(15)

    print(dasps.shape)
    print(sad.shape)

# %%
