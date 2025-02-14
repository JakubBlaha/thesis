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


class DatasetLabel(Enum):
    HI_GAD = 0
    HI_SAD = 1
    LO_GAD = 2
    LO_SAD = 3


class DatasetEnum(Enum):
    DASPS = 0
    SAD = 1


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

    def get_num_classes(self):
        if self.merge_control:
            return 3
        else:
            return 4
        
    def get_label_name(self, label: int):
        if label in [DatasetLabel.HI_GAD.value, DatasetLabel.HI_SAD.value]:
            return DatasetLabel(label).name
        elif label in [DatasetLabel.LO_GAD.value, DatasetLabel.LO_SAD.value] and not self.merge_control:
            return DatasetLabel(label).name
        elif label == DatasetLabel.LO_GAD.value and self.merge_control:
            return "CONTROL"
        else:
            raise ValueError(f'Invalid label: {label}')

    def get_possible_labels(self):
        if self.merge_control:
            return [DatasetLabel.HI_GAD.value, DatasetLabel.HI_SAD.value, DatasetLabel.LO_GAD.value]
        else:
            return [DatasetLabel.HI_GAD.value, DatasetLabel.HI_SAD.value, DatasetLabel.LO_GAD.value, DatasetLabel.LO_SAD.value]


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
    label_counts_before_oversample = label_counts.copy()

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

    # Create a bar chart of the label distribution
    label_counts_after_oversample = np.bincount(labels)

    # Create a side-by-side bar chart of the label distribution before and after oversampling
    # label_names = [DatasetLabel(i).name for i in range(len(label_counts))]

    # x = np.arange(len(label_names))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots(figsize=(12, 6))
    # rects1 = ax.bar(x - width/2, label_counts_before_oversample, width, label='Before Oversampling')
    # rects2 = ax.bar(x + width/2, label_counts_after_oversample, width, label='After Oversampling')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Labels')
    # ax.set_ylabel('Number of samples')
    # ax.set_title('Label Distribution Before and After Oversampling')
    # ax.set_xticks(x, label_names)
    # plt.xticks(rotation=45, ha="right")
    # ax.legend()

    # fig.tight_layout()
    # plt.show()

    return features, labels, groups


def normalize_eeg(data):
    """Normalizes EEG data (samples, electrodes, time_points) channel-wise."""

    # Use axis=2 to normalize across the time points for each channel
    normalized_data = zscore(data, axis=2, ddof=0)  # ddof=0 for population std

    return normalized_data


class DatasetBuilder:
    _feat_names: list[str] = []
    _feat_domain_prefix = ['time', 'abs_pow', 'rel_pow', 'conn', 'ai']

    def __init__(self, labeling_scheme: LabelingScheme, seglen: int) -> None:
        self._labeling_scheme = labeling_scheme
        self.seglen = seglen
        self._preloaded_data = None

    def _drop_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=['dataset', 'ham', 'sam', 'stai', 'subject'])

    def build_dataset_df(self, mode="both",
                         domains:
                         list[str] | None = None,
                         p_val_thresh=0.05) -> pd.DataFrame:
        assert False, "Merge control labels after oversample"

        if domains is None:
            domains = ["time", "rel_pow", "conn", "ai"]

        print("Building dataset for seglen:", self.seglen, "mode:", mode,
              "domains:", domains, "p_val_thresh:", p_val_thresh)

        df = self._get_seglen_df()
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

    def preload_epochs(self, mode='both'):
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

                if mode == 'dasps' and dataset != 'dasps':
                    continue
                elif mode == 'sad' and dataset != 'SAD':
                    continue

                if dataset == 'SAD':
                    label = self._get_sad_label(metadata)
                elif dataset == 'dasps':
                    label = self._get_dasps_label(metadata)
                else:
                    raise ValueError(f'Invalid dataset: {metadata["dataset"]}')

                all_data.append(epoch)
                all_labels.append(label.value)
                all_groups.append(metadata['subject'])

        self._preloaded_data = {
            'data': np.array(all_data),
            'labels': np.array(all_labels),
            'groups': np.array(all_groups)
        }

    def build_deep_datasets_train_test(
            self, *, insert_ch_dim: bool, test_subj_ids: list[int], oversample=True,
            device=None, mode='both'):
        if self._preloaded_data is None:
            self.preload_epochs(mode=mode)

        data = self._preloaded_data['data']
        labels = self._preloaded_data['labels']
        groups = self._preloaded_data['groups']

        test_mask = np.isin(groups, test_subj_ids)

        train_data = data[~test_mask]
        train_labels = labels[~test_mask]

        test_data = data[test_mask]
        test_labels = labels[test_mask]

        train_groups = groups[~test_mask]

        if oversample:
            train_data, train_labels, train_groups = custom_random_oversample(
                train_data, train_labels, train_groups)

        if self._labeling_scheme.merge_control:
            test_labels[test_labels == DatasetLabel.LO_SAD.value] = DatasetLabel.LO_GAD.value
            train_labels[train_labels == DatasetLabel.LO_SAD.value] = DatasetLabel.LO_GAD.value

        # Count values
        train_label_counts = np.bincount(train_labels)
        print("Train label counts:")
        for label, count in enumerate(train_label_counts):
            print(f"Label {label} - {DatasetLabel(label).name}: {count}")

        print()

        test_label_counts = np.bincount(test_labels)
        print("Test label counts:")
        for label, count in enumerate(test_label_counts):
            print(f"Label {label} - {DatasetLabel(label).name}: {count}")

        train_data = normalize_eeg(train_data).astype(np.float32)
        test_data = normalize_eeg(test_data).astype(np.float32)

        train_torch_dataset = TorchDeepDataset(
            train_data, train_labels, insert_ch_dim=insert_ch_dim, device=device)
        test_torch_dataset = TorchDeepDataset(
            test_data, test_labels, insert_ch_dim=insert_ch_dim, device=device)

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

    train, test = builder.build_deep_datasets_train_test(insert_ch_dim=False, test_subj_ids=[1, 2, 3, 102, 103, 104, 401, 403, 405])

# %%
