import mne


class FeatureBase:
    @staticmethod
    def compute(epoch: mne.EpochsArray) -> float:
        pass

# class Hjorth
