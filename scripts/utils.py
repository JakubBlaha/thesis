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
import matplotlib.pyplot as plt

from constants import features_dir

script_path = os.path.dirname(os.path.realpath(__file__))


class DaspsLabeling(Enum):
    HAM = 0
    SAM = 1


class LabelingScheme:
    def __init__(
            self, dasps_labeling: DaspsLabeling, *, lo_level_dasps=[0, 1],
            hi_level_dasps=[2, 3],
            lo_level_sad=[0, 1],
            hi_level_sad=[2, 3],
            merge_control=True) -> None:
        # Make sure low and hi levels do not overlap
        assert len(set(lo_level_dasps) & set(hi_level_dasps)) == 0

        self.dasps_labeling = dasps_labeling
        self.lo_level_dasps = lo_level_dasps
        self.hi_level_dasps = hi_level_dasps
        self.lo_level_sad = lo_level_sad
        self.hi_level_sad = hi_level_sad
        self.merge_control = merge_control


def get_extracted_seglens():
    paths = glob.glob(os.path.join(features_dir, 'features_*s.csv'))
    basenames = [os.path.basename(i) for i in paths]

    seglens = [i.strip("features_").strip("s.csv") for i in basenames]
    seglens = [int(i) for i in seglens]

    return seglens


def get_feats_csv_path(seglen: int):
    return os.path.join(features_dir, f'features_{seglen}s.csv')


def random_oversample(features, labels, groups, oversample_labels=None):
    """Oversample the minority classs to balance the dataset."""

    if oversample_labels is None:
        oversample_labels = np.unique(labels)

    label_counts = np.bincount(labels[np.isin(labels, oversample_labels)])

    print(label_counts)

    max_label_count = np.max(label_counts)

    for label in oversample_labels:
        n_to_oversample = max_label_count - label_counts[label]

        if n_to_oversample == 0:
            continue

        # Get indices of samples with given label
        label_indices = np.where(labels == label)[0]

        # Randomly select samples to oversample
        new_samples = np.random.choice(label_indices, n_to_oversample)

        features = np.vstack([features, features[new_samples]])
        labels = np.hstack([labels, labels[new_samples]])
        groups = np.hstack([groups, groups[new_samples]])

    return features, labels, groups


def normalize_eeg(data):
    """Normalizes EEG data (samples, electrodes, time_points) channel-wise."""

    # Use axis=2 to normalize across the time points for each channel
    normalized_data = zscore(data, axis=2, ddof=0)  # ddof=0 for population std

    return normalized_data


class DatasetBuilder:
    _feat_names: list[str] = []
    _feat_domain_prefix = ['time', 'abs_pow', 'rel_pow', 'conn', 'ai']

    def __init__(self, labeling_scheme: LabelingScheme, seglen: int, mode="both") -> None:
        self._validate_mode(mode)

        self._labeling_scheme = labeling_scheme
        self.seglen = seglen
        self._preloaded_data = None
        self.mode = mode

    def _drop_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['dataset', 'ham', 'sam', 'stai', 'subject'])

    def build_dataset_df(self,
                         domains:
                         list[str] | None = None,
                         p_val_thresh=0.05) -> pd.DataFrame:
        assert False, "Merge control labels after oversample"

        if domains is None:
            domains = ["time", "rel_pow", "conn", "ai"]

        print("Building dataset for seglen:", self.seglen, "mode:", self.mode,
              "domains:", domains, "p_val_thresh:", p_val_thresh)

        df = self._get_seglen_df()
        df = self._label_rows(df)
        df = self._keep_mode_rows(df, self.mode)
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
        low_gad = df[df['label'] == "LO_GAD"].copy()
        gad = df[df['label'] == "HI_GAD"].copy()
        low_sad = df[df['label'] == "LO_SAD"].copy()
        sad = df[df['label'] == "HI_SAD"].copy()

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
    
    def _validate_mode(self, mode: str):
        _valid_modes = ["both", "dasps", "sad"]

        assert mode in _valid_modes, "mode must be one of " + str(
            _valid_modes)

    def _keep_mode_rows(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        self._validate_mode(mode)

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

            df.at[i, 'label'] = label

        return df

    def _get_dasps_label(self, row: pd.Series) -> str:
        if self._labeling_scheme.dasps_labeling.value == DaspsLabeling.HAM.value:
            if row['ham'] in self._labeling_scheme.lo_level_dasps:
                return "LO_GAD"
            elif row['ham'] in self._labeling_scheme.hi_level_dasps:
                return "HI_GAD"
            else:
                raise ValueError(f'Invalid HAM severity: {row["ham"]}')

        elif self._labeling_scheme.dasps_labeling.value == DaspsLabeling.SAM.value:
            if row['sam'] in self._labeling_scheme.lo_level_dasps:
                return "LO_GAD"
            elif row['sam'] in self._labeling_scheme.hi_level_dasps:
                return "HI_GAD"
            else:
                raise ValueError(f'Invalid SAM severity: {row["sam"]}')

        raise ValueError('Invalid labeling scheme')

    def _get_sad_label(self, row: pd.Series) -> str:
        severity = row['stai']

        if severity in self._labeling_scheme.lo_level_sad:
            return "LO_SAD"
        elif severity in self._labeling_scheme.hi_level_sad:
            return "HI_SAD"

        raise ValueError(f'Invalid SAD severity: {severity}')
    
    def _get_segment_files(self):
        clean_segdir_path = os.path.join(
            script_path, f'../data/segmented/{self.seglen}s/clean')
        files = glob.glob(f'{clean_segdir_path}/*-epo.fif')
        files = sorted(files)

        return files
    
    def _get_subj_id_from_path(self, path: str) -> int:
        return int(os.path.basename(path).strip('-epo.fif').strip('S'))

    def get_subj_ids(self) -> list[int]:
        files = self._get_segment_files()
        subj_ids = [self._get_subj_id_from_path(f) for f in files]

        return subj_ids

    def preload_epochs(self):
        """Preloads all epochs into memory."""
        if self._preloaded_data is not None:
            print("Epochs already preloaded. Skipping preload.")
            return

        files = self._get_segment_files()
        all_data = []
        all_labels = []
        all_groups = []

        for f in files:
            epochs = mne.read_epochs(f, preload=True, verbose=False)

            for index, epoch in enumerate(epochs):
                metadata = epochs.metadata.iloc[index]
                dataset = metadata['dataset']

                if self.mode == 'dasps' and dataset != 'dasps':
                    continue
                elif self.mode == 'sad' and dataset != 'SAD':
                    continue

                if dataset == 'SAD':
                    label = self._get_sad_label(metadata)
                elif dataset == 'dasps':
                    label = self._get_dasps_label(metadata)
                else:
                    raise ValueError(f'Invalid dataset: {metadata["dataset"]}')

                all_data.append(epoch)
                all_labels.append(label)
                all_groups.append(metadata['subject'])

        self._preloaded_data = {
            'data': np.array(all_data),
            'labels': np.array(all_labels),
            'groups': np.array(all_groups)
        }

    def _output_label_counts(self, labels, label_dict=None):
        label_counts = np.bincount(labels)

        print("Label counts:")
        for label, count in enumerate(label_counts):
            label_name = label_dict[label] if label_dict is not None else label
            print(f"{label_name}: {count}")

    def build_deep_datasets_train_test(
            self, *, insert_ch_dim: bool, test_subj_ids: list[int], oversample=True,
            device=None):
        if self._preloaded_data is None:
            self.preload_epochs()

        data = self._preloaded_data['data']
        labels = self._preloaded_data['labels']
        groups = self._preloaded_data['groups']

        test_mask = np.isin(groups, test_subj_ids)

        train_data = data[~test_mask]
        train_labels = labels[~test_mask]

        test_data = data[test_mask]
        test_labels = labels[test_mask]

        train_groups = groups[~test_mask]

        # Encode labels
        label_to_int = {label: i for i, label in enumerate(np.unique(train_labels))}
        int_to_label = {i: label for label, i in label_to_int.items()}

        train_labels = np.array([label_to_int[label] for label in train_labels])
        test_labels = np.array([label_to_int[label] for label in test_labels])

        # self._output_label_counts(train_labels, int_to_label)

        if oversample:
            if self._labeling_scheme.merge_control and self.mode == "both":
                train_data, train_labels, train_groups = random_oversample(
                    train_data, train_labels, train_groups, oversample_labels=[label_to_int["LO_GAD"], label_to_int["LO_SAD"]])
            
            # self._output_label_counts(train_labels, int_to_label)

            train_data, train_labels, train_groups = random_oversample(
                train_data, train_labels, train_groups)

        train_data = normalize_eeg(train_data).astype(np.float32)
        test_data = normalize_eeg(test_data).astype(np.float32)

        if self._labeling_scheme.merge_control and self.mode == "both":
            int_lo_gad = label_to_int["LO_GAD"]
            int_lo_sad = label_to_int["LO_SAD"]

            int_control = min(int_lo_gad, int_lo_sad)
            int_to_label[int_control] = "CONTROL"

            train_labels[(train_labels == int_lo_gad) | (train_labels == int_lo_sad)] = int_control
            test_labels[(test_labels == int_lo_gad) | (test_labels == int_lo_sad)] = int_control

        # self._output_label_counts(train_labels, int_to_label)

        train_torch_dataset = TorchDeepDataset(
            train_data, train_labels, insert_ch_dim=insert_ch_dim, device=device)
        test_torch_dataset = TorchDeepDataset(
            test_data, test_labels, insert_ch_dim=insert_ch_dim, device=device)
        
        self.last_int_to_label = int_to_label

        return train_torch_dataset, test_torch_dataset


class TorchDeepDataset(Dataset):
    def __init__(self, data, labels, insert_ch_dim: bool, device=None) -> None:
        # Add channel dimension
        if insert_ch_dim:
            data = np.array([i[np.newaxis, :, :] for i in data])
        else:
            data = np.array(data)

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
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM, merge_control=True)
    builder = DatasetBuilder(labeling_scheme, seglen=3)

    train, test = builder.build_deep_datasets_train_test(insert_ch_dim=False, test_subj_ids=[101, 102, 103], oversample=True)

# %%
