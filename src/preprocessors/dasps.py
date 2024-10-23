from dataclasses import dataclass
from enum import Enum
from autoreject import AutoReject
import csv
import h5py
import numpy as np
import mne
import os
import matplotlib.pyplot as plt

from common import CHANNEL_NAMES, Trial, TrialLabel


DASPS_FS = 128
DATA_PATH = os.path.abspath("../data/dasps")

# Subjects are split into two groups based on HAM-A score at the end of the
# experiment. Scores < 20 (normal, light) are classified as low anxiety, scores > 20
# (moderate, severe) are classified as high anxiety. There are no scores equal to 20.
HAM_HA_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 16, 17, 19, 20, 21, 22]
HAM_LA_SUBJECTS = [8, 9, 10, 14, 15, 18, 23]


def get_subject_epochs(subject_id: int):
    fname = os.path.join(DATA_PATH, 'raw_mat', 'S' + str(subject_id).zfill(2) + ".mat")

    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])

    channel_types = ['eeg'] * len(CHANNEL_NAMES)

    info = mne.create_info(
        ch_names=CHANNEL_NAMES, ch_types=channel_types, sfreq=DASPS_FS)  # type: ignore
    info.set_montage('standard_1020')

    print(data.shape)

    data = data.swapaxes(1, 2)

    epochs = mne.EpochsArray(data, info)

    return epochs


class Severity(Enum):
    NORMAL = 0
    LIGHT = 1
    MODERATE = 2
    SEVERE = 3


@dataclass
class DaspsTrial:
    subject_id: int
    index: int
    valence: float
    arousal: float

    sam_label: Severity | None = None

    def __post_init__(self):
        # Calculate anxiety severity from valence and arousal, e.g. the SAM label
        if self.valence > 5 or self.arousal < 5:
            self.sam_label = Severity.NORMAL
        elif self.valence in [0, 1, 2] and self.arousal in [7, 8, 9]:
            self.sam_label = Severity.SEVERE
        elif self.valence in [2, 3, 4] and self.arousal in [6]:
            self.sam_label = Severity.MODERATE
        elif self.valence in [4, 5] and self.arousal in [5]:
            self.sam_label = Severity.LIGHT
        else:
            self.sam_label = Severity.NORMAL

    def get_epochs_precleaned(self, duration: int):
        # fname = os.path.join(DATA_PATH, 'preprocessed', self.subject_id + "preprocessed.mat")
        fname = os.path.join(DATA_PATH, 'raw_mat', 'S' + str(self.subject_id).zfill(2) + ".mat")

        with h5py.File(fname, 'r') as f:
            data = np.array(f['data'])

        print(data.shape)

        recitation_epoch_idx = self.index * 2
        imagining_epoch_idx = recitation_epoch_idx + 1

        sit_epochs = data[recitation_epoch_idx:imagining_epoch_idx + 1]
        sit_epochs = sit_epochs.swapaxes(1, 2)

        channel_types = ['eeg'] * len(CHANNEL_NAMES)  # All channels are EEG
        info = mne.create_info(
            ch_names=CHANNEL_NAMES, ch_types=channel_types, sfreq=DASPS_FS)  # type: ignore
        info.set_montage('standard_1020')

        epochs = mne.EpochsArray(sit_epochs, info)

        # Preprocess
        epochs.filter(l_freq=0.5, h_freq=30)
        ica = mne.preprocessing.ICA(n_components=14, random_state=97, max_iter=800)
        ica.fit(epochs)

        # ica.plot_components()

        # epochs.plot_psd()

        data = epochs.get_data()

        first = data[0]
        second = data[1]

        both = np.concatenate((first, second), axis=1)
        both = mne.io.RawArray(both, info)

        epochs = mne.make_fixed_length_epochs(both, duration=duration, overlap=0.5 * duration * 0)

        return epochs


class DaspsSubject:
    subject_id: int

    def __init__(self, subject_id: int):
        self.subject_id = subject_id

    def get_epochs_custom_cleaned(self):
        pass

        # fname = os.path.join(DATA_PATH, 'raw_edf', 'S' + str(self.subject_id).zfill(2) + ".edf")

        # raw = mne.io.read_raw_edf(fname, preload=True)
        # raw = raw.filter(l_freq=0.5, h_freq=40, picks=CHANNEL_NAMES)

        # print(raw.info)

        # raw = raw.pick(CHANNEL_NAMES)

        # raw.plot(n_channels=14, duration=30, scalings=200e-6)
        # plt.show()
        # return raw

    def get_epochs(self):
        epochs = get_subject_epochs(self.subject_id)

        return epochs


class DaspsPreprocessor:
    @staticmethod
    def get_trials() -> list[DaspsTrial]:
        situations: list[DaspsTrial] = []

        with open('../data/DASPS.csv') as f:
            reader = csv.DictReader(f)

            for row in reader:
                _stripped = row['Id Participant'].strip('S')

                subject_id = int(_stripped) if _stripped else situations[-1].subject_id
                sit_index = int(row['Id situation ']) - 1
                valence = int(row['valence'])
                arousal = int(row['Arousal'])

                situation = DaspsTrial(
                    subject_id, sit_index, valence, arousal)

                situations.append(situation)

        return situations

    @staticmethod
    def get_ha_sam_trials() -> list[DaspsTrial]:
        return [i for i in DaspsPreprocessor.get_trials() if i.sam_label in [Severity.SEVERE, Severity.MODERATE]]

    @staticmethod
    def get_la_sam_trials() -> list[DaspsTrial]:
        return [i for i in DaspsPreprocessor.get_trials() if i.sam_label in [Severity.NORMAL]]

    @staticmethod
    def get_ha_ham_subjects() -> list[DaspsSubject]:
        return [DaspsSubject(i) for i in HAM_HA_SUBJECTS]

    @staticmethod
    def get_la_ham_subjects() -> list[DaspsSubject]:
        return [DaspsSubject(i) for i in HAM_LA_SUBJECTS]

    # @staticmethod
    # def _get_normal_epochs(duration, autoreject=False) -> mne.EpochsArray:
    #     normal_sits = DaspsPreprocessor.get_normal_sits()

    #     epochs = mne.concatenate_epochs(
    #         [sit.get_epoch_array(duration) for sit in normal_sits])

    #     if autoreject:
    #         ar = AutoReject()
    #         epochs = ar.fit_transform(epochs)

    #     return epochs

    # @staticmethod
    # def _get_severe_moderate_epochs(duration, autoreject=False) -> mne.EpochsArray:
    #     severe_moderate_sits = DaspsPreprocessor.get_severe_moderate_sits()

    #     epochs = mne.concatenate_epochs(
    #         [sit.get_epoch_array(duration) for sit in severe_moderate_sits])

    #     if autoreject:
    #         ar = AutoReject()
    #         epochs = ar.fit_transform(epochs)

    #     return epochs

    # @staticmethod
    # def _get_interval_from_seconds(seconds):
    #     max_seconds = 14.9
    #     default_start_seconds = 7

    #     optimal_end_time = default_start_seconds + seconds
    #     missing_seconds = optimal_end_time - max_seconds

    #     move_start_left = max(0, missing_seconds)

    #     seconds_start = default_start_seconds - move_start_left
    #     seconds_end = seconds_start + seconds

    #     assert seconds_end <= max_seconds

    #     return seconds_start, seconds_end

    @staticmethod
    def get_trials_(duration: int, autoreject=False):
        normal_epochs = DaspsPreprocessor._get_normal_epochs(duration, autoreject=autoreject)
        anx_epochs = DaspsPreprocessor._get_severe_moderate_epochs(duration, autoreject=autoreject)

        # _trial_w = DaspsPreprocessor._get_interval_from_seconds(trial_seconds)

        normal_trials = [Trial(TrialLabel.CONTROL, normal_epochs[i])
                         for i in range(len(normal_epochs))]
        anx_trials = [Trial(TrialLabel.GAD, anx_epochs[i])
                      for i in range(len(anx_epochs))]

        return normal_trials + anx_trials
