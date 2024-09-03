from dataclasses import dataclass
from enum import Enum
import csv
import h5py
import numpy as np
import mne


DASPS_FS = 128
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7',
                 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


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

    def get_both_epochs(self):
        # filename = f'data/dasps_preprocessed/{self.id_}preprocessed.mat'
        filename = f'data/dasps_raw_mat/{self.id_}.mat'

        print(f"Getting epochs for filename {filename}")

        with h5py.File(filename, 'r') as f:
            data = np.array(f['data'])

        recitation_epoch_idx = self.index * 2
        imagining_epoch_idx = recitation_epoch_idx + 1

        sit_epochs = data[recitation_epoch_idx:imagining_epoch_idx + 1]
        sit_epochs = sit_epochs.swapaxes(1, 2)

        # print(sit_epochs.shape)
        # print(recitation_epoch_idx, imagining_epoch_idx)

        channel_types = ['eeg'] * len(CHANNEL_NAMES)  # All channels are EEG
        info = mne.create_info(
            ch_names=CHANNEL_NAMES, ch_types=channel_types, sfreq=DASPS_FS)  # type: ignore
        epochs = mne.EpochsArray(sit_epochs, info)

        # print(epochs)
        # print(epochs.info)

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
    def get_normal_epochs() -> mne.EpochsArray:
        normal_sits = DaspsPreprocessor.get_normal_sits()

        normal_epochs = mne.concatenate_epochs(
            [sit.get_both_epochs() for sit in normal_sits])

        return normal_epochs

    @staticmethod
    def get_severe_moderate_epochs() -> mne.EpochsArray:
        severe_moderate_sits = DaspsPreprocessor.get_severe_moderate_sits()

        severe_moderate_epochs = mne.concatenate_epochs(
            [sit.get_both_epochs() for sit in severe_moderate_sits])

        return severe_moderate_epochs


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
