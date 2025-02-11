# %%
import mne
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from enum import Enum
from scipy.stats import f_oneway
from scipy.stats import zscore
import torch

from constants import features_dir

script_path = os.path.dirname(os.path.realpath(__file__))


class DaspsLabeling(Enum):
    HAM = 0
    SAM = 1


class DatasetLabel(Enum):
    CONTROL = 0
    HI_GAD = 1
    HI_SAD = 2
    LO_GAD = 3
    LO_SAD = 4


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


# def custom_random_oversample(features, labels, groups):
#     # Replace LO_GAD and LO_SAD labels with CONTROL
#     labels[labels == DatasetLabel.LO_GAD.value] = DatasetLabel.CONTROL.value
#     labels[labels == DatasetLabel.LO_SAD.value] = DatasetLabel.CONTROL.value

#     label_counts = np.bincount(labels)
#     max_label_count = np.max(label_counts)

#     print("Label counts before oversampling:")
#     for label, count in enumerate(label_counts):
#         print(f"Label {label} - {DatasetLabel(label).name}: {count}")

#     for label in np.unique(labels):
#         n_to_oversample = max_label_count - label_counts[label]

#         if n_to_oversample == 0:
#             continue

#         # Get indices of samples with given label
#         label_indices = np.where(labels == label)[0]

#         # Randomly select samples to oversample
#         new_samples = np.random.choice(label_indices, n_to_oversample)

#         features = np.vstack([features, features[new_samples]])
#         labels = np.hstack([labels, labels[new_samples]])
#         groups = np.hstack([groups, groups[new_samples]])

#     return features, labels, groups

def custom_random_oversample(features, labels, groups):
    label_counts = np.bincount(labels)
    print("Label counts before oversampling:")
    for label, count in enumerate(label_counts):
        print(f"Label {label} - {DatasetLabel(label).name}: {count}")

    # Balance LO_GAD and LO_SAD
    lo_gad_count = label_counts[DatasetLabel.LO_GAD.value]
    lo_sad_count = label_counts[DatasetLabel.LO_SAD.value]

    if lo_gad_count > lo_sad_count:
        label_to_oversample = DatasetLabel.LO_SAD.value
        n_to_oversample = lo_gad_count - lo_sad_count
    elif lo_sad_count > lo_gad_count:
        label_to_oversample = DatasetLabel.LO_GAD.value
        n_to_oversample = lo_sad_count - lo_gad_count
    else:
        label_to_oversample = None
        n_to_oversample = 0

    if label_to_oversample is not None:
        label_indices = np.where(labels == label_to_oversample)[0]
        if len(label_indices) > 0:
            new_samples = np.random.choice(label_indices, n_to_oversample)
            features = np.vstack([features, features[new_samples]])
            labels = np.hstack([labels, labels[new_samples]])
            groups = np.hstack([groups, groups[new_samples]])
        else:
            print(f"Warning: No samples found for label {label_to_oversample}")

    # Get new label counts after balancing LO_GAD and LO_SAD
    label_counts = np.bincount(labels)
    lo_gad_count = label_counts[DatasetLabel.LO_GAD.value]
    lo_sad_count = label_counts[DatasetLabel.LO_SAD.value]
    
    # Calculate target count for HI_GAD and HI_SAD
    target_count = lo_gad_count + lo_sad_count

    # Oversample HI_GAD and HI_SAD to match target_count
    for label in [DatasetLabel.HI_GAD.value, DatasetLabel.HI_SAD.value]:
        label_count = label_counts[label]
        n_to_oversample = target_count - label_count

        if n_to_oversample > 0:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > 0:
                new_samples = np.random.choice(label_indices, n_to_oversample)
                features = np.vstack([features, features[new_samples]])
                labels = np.hstack([labels, labels[new_samples]])
                groups = np.hstack([groups, groups[new_samples]])
            else:
                print(f"Warning: No samples found for label {label}")

    print("Label counts after oversampling:")
    label_counts = np.bincount(labels)
    for label, count in enumerate(label_counts):
        print(f"Label {label} - {DatasetLabel(label).name}: {count}")

    return features, labels, groups


def normalize_eeg(data):
    """Normalizes EEG data (samples, electrodes, time_points) channel-wise."""

    # Use axis=2 to normalize across the time points for each channel
    normalized_data = zscore(data, axis=2, ddof=0)  # ddof=0 for population std

    return normalized_data


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


class DatasetBuilder(BaseDatasetBuilder):
    _feat_names: list[str] = []
    _feat_domain_prefix = ['time', 'abs_pow', 'rel_pow', 'conn', 'ai']

    def __init__(self, labeling_scheme: LabelingScheme) -> None:
        self._labeling_scheme = labeling_scheme

    def get_uniq_subj_ids(self, seglen: int) -> list[int]:
        df = self._get_seglen_df(seglen)
        return df['uniq_subject_id'].unique().tolist()

    def build_dataset_df(self, seglen: int, mode="both",
                         domains:
                         list[str] | None = None,
                         p_val_thresh=0.05) -> pd.DataFrame:
        
        assert False, "Merge control labels after oversample"

        if domains is None:
            domains = ["time", "rel_pow", "conn", "ai"]

        print("Building dataset for seglen:", seglen, "mode:", mode,
              "domains:", domains, "p_val_thresh:", p_val_thresh)

        df = self._get_seglen_df(seglen)
        df = self._label_rows(df)
        df = self._keep_mode_rows(df, mode)
        df = self._drop_redundant_columns(df)
        df = self._keep_feat_cols(df, domains)
        df = self._keep_significant_cols(df, p_val_thresh)

        self._feat_names = df.columns.tolist()[:-2]

        return df

    def get_feat_names(self):
        return self._feat_names

    def _keep_feat_cols(
            self, df: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
        print("Keeping domains:", domains)

        for d in domains:
            if d not in self._feat_domain_prefix:
                raise ValueError(f'Invalid domain: {d}')

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
        low_gad = df[df['label'] == DatasetLabel.LO_GAD.value].copy()
        gad = df[df['label'] == DatasetLabel.HI_GAD.value].copy()
        low_sad = df[df['label'] == DatasetLabel.LO_SAD.value].copy()
        sad = df[df['label'] == DatasetLabel.HI_SAD.value].copy()

        low_anxiety = pd.concat([low_gad, low_sad])

        for group in [low_anxiety, gad, sad]:
            group.drop(columns=['label', 'uniq_subject_id'],
                       inplace=True)

        p_vals = {}

        nonempty_groups = [i for i in [low_anxiety, gad, sad] if not i.empty]

        # Use low_anxiety columns as reference
        for col in low_anxiety.columns:
            f_val, p_val = f_oneway(*[group[col] for group in nonempty_groups])
            p_vals[col] = p_val

        # Use low_anxiety columns as reference
        for col in low_anxiety.columns:
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
                return DatasetLabel.LO_GAD
            elif row['ham'] in self._labeling_scheme.hi_level_dasps:
                return DatasetLabel.HI_GAD
            else:
                raise ValueError(f'Invalid HAM severity: {row["ham"]}')

        elif self._labeling_scheme.dasps_labeling.value == DaspsLabeling.SAM.value:
            if row['sam'] in self._labeling_scheme.lo_level_dasps:
                return DatasetLabel.LO_GAD
            elif row['sam'] in self._labeling_scheme.hi_level_dasps:
                return DatasetLabel.HI_GAD
            else:
                raise ValueError(f'Invalid SAM severity: {row["sam"]}')

        raise ValueError('Invalid labeling scheme')

    def _get_sad_label(self, row: pd.Series) -> DatasetLabel:
        severity = row['stai']

        if severity in self._labeling_scheme.lo_level_sad:
            return DatasetLabel.LO_SAD
        elif severity in self._labeling_scheme.hi_level_sad:
            return DatasetLabel.HI_SAD

        raise ValueError(f'Invalid SAD severity: {severity}')

    def build_deep_datasets_train_test(
            self, *, seglen: int, insert_ch_dim: bool, test_subj_ids: list[int], oversample=True,
            device=None):
        clean_segdir_path = os.path.join(
            script_path, f'../data/segmented/{seglen}s/clean')
        files = glob.glob(f'{clean_segdir_path}/*-epo.fif')
        files = sorted(files)

        data = []
        labels = []
        groups = []

        for f in files:
            epochs = mne.read_epochs(f, preload=True, verbose=False)

            for index, epoch in enumerate(epochs):
                metadata = epochs.metadata.iloc[index]

                if metadata['dataset'] == 'SAD':
                    label = self._get_sad_label(metadata)
                elif metadata['dataset'] == 'dasps':
                    label = self._get_dasps_label(metadata)
                else:
                    raise ValueError(f'Invalid dataset: {metadata["dataset"]}')

                data.append(epoch)
                labels.append(label.value)
                groups.append(metadata['subject'])

        data = np.array(data)
        labels = np.array(labels)
        groups = np.array(groups)

        data = normalize_eeg(data).astype(np.float32)

        unique_groups = np.unique(groups)
        print("Available subject IDs:", unique_groups)
        print("Test subject IDs:", test_subj_ids)

        if oversample:
            data, labels, groups = custom_random_oversample(
                data, labels, groups)
            
        # Replace LO_GAD and LO_SAD labels with CONTROL
        labels[labels == DatasetLabel.LO_GAD.value] = DatasetLabel.CONTROL.value
        labels[labels == DatasetLabel.LO_SAD.value] = DatasetLabel.CONTROL.value

        test_mask = np.isin(groups, test_subj_ids)

        train_data = data[~test_mask]
        train_labels = labels[~test_mask]

        test_data = data[test_mask]
        test_labels = labels[test_mask]

        train_torch_dataset = TorchDeepDataset(
            train_data, train_labels, insert_ch_dim=insert_ch_dim, device=device)
        test_torch_dataset = TorchDeepDataset(
            test_data, test_labels, insert_ch_dim=insert_ch_dim, device=device)

        return train_torch_dataset, test_torch_dataset


class TorchDeepDataset(Dataset):
    def __init__(self, data, labels, insert_ch_dim: bool, device=None) -> None:
        self.max_len = 1024

        # Add channel dimension
        if insert_ch_dim:
            data = np.array([i[np.newaxis, :, :] for i in data])
        else:
            data = np.array(data)

        print("Shape", data.shape)

        self.epochs = torch.from_numpy(data).float().to(device)
        self.labels = torch.from_numpy(labels).long().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.epochs[index], self.labels[index]


if __name__ == "__main__":
    # Normal dataset builder
    # labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    # builder = DatasetBuilder(labeling_scheme)

    # df = builder.build_dataset_df(10)

    # print(dasps.shape)
    # print(sad.shape)

    # Deep dataset builder
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    train, test = builder.build_deep_datasets_train_test(seglen=10, insert_ch_dim=False, test_subj_ids=[1, 2, 3])

# %%
