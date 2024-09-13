from dataclasses import dataclass
from enum import Enum
from autoreject import AutoReject
import csv
import h5py
import numpy as np
import mne

from common import CHANNEL_NAMES, Trial, TrialLabel


DASPS_FS = 128


class Severity(Enum):
    NORMAL = 0
    LIGHT = 1
    MODERATE = 2
    SEVERE = 3


class DaspsTrialKind(Enum):
    RECITING = 0
    IMAGINING = 1


@dataclass
class DaspsTrial:
    kind: DaspsTrialKind
    recording = None


@dataclass
class DaspsSituation:
    id_: str
    index: int
    valence: float
    arousal: float

    def get_sam_label(self):
        if self.valence > 5 or self.arousal < 5:
            # print('normal 1')
            return Severity.NORMAL

        if self.valence in [0, 1, 2] and self.arousal in [7, 8, 9]:
            # print('severe')
            return Severity.SEVERE

        if self.valence in [2, 3, 4] and self.arousal in [6]:
            # print('moderate')
            return Severity.MODERATE

        if self.valence in [4, 5] and self.arousal in [5]:
            # print('light')
            return Severity.LIGHT

        # print('normal 2')
        return Severity.NORMAL

    # def get_trials(self):
    #     filename = f'data/dasps_raw_edf/{self.id_}.edf'

    #     raw = mne.io.read_raw_edf(filename, preload=True)

    #     print(raw.info)

    #     raw = raw.pick(CHANNEL_NAMES)

    #     raw.plot(n_channels=14, duration=30, scalings=200e-6)
    #     plt.show()

    def get_epoch_array(self, duration: int):
        # filename = f'data/dasps_preprocessed/{self.id_}preprocessed.mat'
        filename = f'data/dasps_raw_mat/{self.id_}.mat'

        print(f"Getting epochs for filename {filename}")

        with h5py.File(filename, 'r') as f:
            data = np.array(f['data'])

        recitation_epoch_idx = self.index * 2
        imagining_epoch_idx = recitation_epoch_idx + 1

        sit_epochs = data[recitation_epoch_idx:imagining_epoch_idx + 1]
        sit_epochs = sit_epochs.swapaxes(1, 2)

        channel_types = ['eeg'] * len(CHANNEL_NAMES)  # All channels are EEG
        info = mne.create_info(
            ch_names=CHANNEL_NAMES, ch_types=channel_types, sfreq=DASPS_FS)  # type: ignore
        info.set_montage('standard_1020')

        epochs = mne.EpochsArray(sit_epochs, info)

        data = epochs.get_data()

        # print(data.shape)

        first = data[0]
        second = data[1]

        both = np.concatenate((first, second), axis=1)
        both = mne.io.RawArray(both, info)

        epochs = mne.make_fixed_length_epochs(both, duration=duration, overlap=0.5 * duration)

        return epochs


class DaspsPreprocessor:
    @staticmethod
    def get_situations() -> list[DaspsSituation]:
        situations: list[DaspsSituation] = []

        with open('data/DASPS.csv') as f:
            reader = csv.DictReader(f)

            for row in reader:
                subject_id = row['Id Participant'] or situations[-1].id_
                sit_index = int(row['Id situation ']) - 1
                valence = int(row['valence'])
                arousal = int(row['Arousal'])

                situation = DaspsSituation(
                    subject_id, sit_index, valence, arousal)

                situations.append(situation)

        return situations

    @staticmethod
    def get_severe_moderate_sits() -> list[DaspsSituation]:
        situations = DaspsPreprocessor.get_situations()

        severe_moderate_situations = []

        for sit in situations:
            label = sit.get_sam_label()

            if label in [Severity.SEVERE, Severity.MODERATE]:
                severe_moderate_situations.append(sit)

        # print(list(map(lambda x: x.id_, severe_moderate_situations)))

        return severe_moderate_situations

    @staticmethod
    def get_normal_sits() -> list[DaspsSituation]:
        situations = DaspsPreprocessor.get_situations()

        normal_situations = []

        for sit in situations:
            label = sit.get_sam_label()

            if label == Severity.NORMAL:
                normal_situations.append(sit)

        # print(list(map(lambda x: x.id_, severe_moderate_situations)))

        return normal_situations

    @staticmethod
    def _get_normal_epochs(duration, autoreject=False) -> mne.EpochsArray:
        normal_sits = DaspsPreprocessor.get_normal_sits()

        epochs = mne.concatenate_epochs(
            [sit.get_epoch_array(duration) for sit in normal_sits])

        if autoreject:
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)

        return epochs

    @staticmethod
    def _get_severe_moderate_epochs(duration, autoreject=False) -> mne.EpochsArray:
        severe_moderate_sits = DaspsPreprocessor.get_severe_moderate_sits()

        epochs = mne.concatenate_epochs(
            [sit.get_epoch_array(duration) for sit in severe_moderate_sits])

        if autoreject:
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)

        return epochs

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
    def get_trials(duration: int, autoreject=False):
        normal_epochs = DaspsPreprocessor._get_normal_epochs(duration, autoreject=autoreject)
        anx_epochs = DaspsPreprocessor._get_severe_moderate_epochs(duration, autoreject=autoreject)

        # _trial_w = DaspsPreprocessor._get_interval_from_seconds(trial_seconds)

        normal_trials = [Trial(TrialLabel.CONTROL, normal_epochs[i])
                         for i in range(len(normal_epochs))]
        anx_trials = [Trial(TrialLabel.GAD, anx_epochs[i])
                      for i in range(len(anx_epochs))]

        return normal_trials + anx_trials


if __name__ == '__main__':
    # severe_moderate_sits = DaspsPreprocessor.get_severe_moderate_sits()
    # normal_sits = DaspsPreprocessor.get_normal_sits()

    # severe_moderate_epochs = mne.concatenate_epochs(
    #     [sit.get_both_epochs() for sit in severe_moderate_sits])

    # normal_epochs = mne.concatenate_epochs(
    #     [sits.get_both_epochs() for sits in normal_sits])

    # # severe_moderate_epochs.compute_psd().plot()
    # # normal_epochs.compute_psd().plot()

    # # print(len(normal_epochs))
    # # print(len(severe_moderate_epochs))

    # # severe_moderate_epochs.compute_tfr('morlet', np.arange(1, 30, 5))[0].plot()

    # plt.show()
    pass
