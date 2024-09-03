from enum import Enum
import mne
import mne_connectivity
import mne_connectivity.viz
import mne_features
import numpy as np
from mne_features import feature_extraction, univariate
from dataclasses import dataclass
import matplotlib.pyplot as plt

from scripts.dasps_preprocess import DaspsPreprocessor, CHANNEL_NAMES


sfreq = 128
n_channels = 14
freq_bands = {
    # 'delta': (0.5, 4), # delta has been filtered out
    'theta': [4, 8],
    'alpha-1': [8, 10],
    'alpha-2': [10, 13],
    'beta': [13, 30],
    'gamma': [30, 45],  # above 45 has been filtered out
}
freq_band_names = list(freq_bands.keys())
n_bands = len(freq_bands)
min_freqs = [freq_bands[band][0] for band in freq_bands]
max_freqs = [freq_bands[band][1] for band in freq_bands]
left_channels = sorted([i for i in CHANNEL_NAMES if int(i[-1]) % 2 == 1])
right_channels = sorted([i for i in CHANNEL_NAMES if int(i[-1]) % 2 == 0])


class TrialLabel(Enum):
    CONTROL = 0
    GAD = 1
    SAD = 2


class Trial:
    def __init__(self, trial_label: TrialLabel, epoch: mne.EpochsArray, crop_start_end=None) -> None:
        self.trial_label = trial_label

        if crop_start_end:
            self.epoch = epoch.crop(tmin=crop_start_end[0], tmax=crop_start_end[1])
        else:
            self.epoch = epoch

        self.info = mne.create_info(
            CHANNEL_NAMES,
            sfreq,
            ch_types='eeg',
        )
        self.info.set_montage('standard_1020')
        self.features = {}

    # def show_spectrogram(self):
    #     freqs = np.logspace(*np.log10([6, 35]), num=8)
    #     n_cycles = freqs / 2

    #     power = self.epoch.compute_tfr(
    #         method="morlet",
    #         freqs=freqs,
    #         n_cycles=n_cycles,
    #         average=True,
    #         decim=3
    #     )

    #     power.plot(baseline=(-0.5, 0), mode='logratio',
    #                title='Average power')

    def compute_time_measures(self):
        # Hurst exp: long term memory, positive follows positive, negative follows negative
        # Approximate entropy:

        funcs = ['hjorth_mobility', 'hjorth_complexity', 'variance', 'app_entropy', 'samp_entropy',
                 'higuchi_fd', 'katz_fd', 'line_length', 'skewness', 'kurtosis', 'rms', 'hurst_exp', 'decorr_time']
        # extractor = feature_extraction.FeatureExtractor(sfreq, funcs)
        data = self.epoch.get_data()

        # for band_name, (l_freq, h_freq) in freq_bands.items():
        #     filtered_data = mne.filter.filter_data(data, sfreq, l_freq, h_freq)

        x = mne_features.feature_extraction.extract_features(data, sfreq, funcs, n_jobs=1)
        x = x.reshape((len(funcs), n_channels))

        for func_name, channels in zip(funcs, x):
            for ch_val, ch_name in zip(channels, CHANNEL_NAMES):
                feat_name = f'time_{func_name}_{ch_name}'
                self.features[feat_name] = ch_val

    def compute_powers(self, normalize):
        # 70 values at the start are powers
        freq_bands_ = np.asanyarray([freq_bands[band][0]
                                     for band in freq_bands] + [freq_bands['gamma'][1]])
        print(freq_bands_)
        print(freq_bands_.shape)

        powers_and_ratios = univariate.compute_pow_freq_bands(
            sfreq, self.epoch.get_data()[0],
            freq_bands_, normalize=normalize, ratios='all', psd_method='welch',
            psd_params={'welch_n_overlap': sfreq // 2})
        powers = powers_and_ratios[:n_channels *
                                   n_bands].reshape((n_channels, -1))

        for el_idx, el in enumerate(powers):
            for band_idx, band_pow in enumerate(el):
                feat_name = f'{'rel' if normalize else 'abs'}_pow_{freq_band_names[band_idx]}_{
                    CHANNEL_NAMES[el_idx]}'
                self.features[feat_name] = band_pow

    def compute_connectivity(self):
        min_freq = min(min_freqs)
        max_freq = max(max_freqs)

        freqs = np.linspace(min_freq, max_freq, int(
            (max_freq - min_freq) * 4 + 1))

        # print(freqs.shape)
        res = mne_connectivity.spectral_connectivity_time(
            self.epoch, freqs=freqs, method="pli", sfreq=sfreq, mode="cwt_morlet", fmin=min_freqs, fmax=max_freqs,
            faverage=True).get_data()

        conn_of_one_epoch = res[0]
        matrix = conn_of_one_epoch.reshape(
            (n_channels, n_channels, n_bands))

        matrix = np.moveaxis(matrix, 2, 0)

        for band_idx, band in enumerate(matrix):
            for el_idx, el in enumerate(band):
                for el2_idx, el2 in enumerate(el[:el_idx]):
                    feat_name = f'conn_{freq_band_names[band_idx]}_{
                        CHANNEL_NAMES[el_idx]}_{CHANNEL_NAMES[el2_idx]}'
                    self.features[feat_name] = el2

        return matrix

    def compute_asymmetry(self):
        for band_name in freq_band_names:
            for left_ch_name, right_ch_name in zip(left_channels, right_channels):
                left_abs = self.features['abs_pow_' + band_name + '_' + left_ch_name]
                right_abs = self.features['abs_pow_' + band_name + '_' + right_ch_name]

                ai = np.log(right_abs) - np.log(left_abs)

                self.features[f'ai_{band_name}_{right_ch_name}-{left_ch_name}'] = ai

    def compute_all_features(self):
        self.compute_time_measures()
        self.compute_powers(normalize=True)
        self.compute_powers(normalize=False)
        self.compute_connectivity()
        self.compute_asymmetry()


def get_trials_and_labels(trial_kwargs={}):
    normal_epochs = DaspsPreprocessor.get_normal_epochs()
    anx_epochs = DaspsPreprocessor.get_severe_moderate_epochs()

    all_epochs = []

    for i, _ in enumerate(normal_epochs):
        trial = Trial(TrialLabel.CONTROL, normal_epochs[i], **trial_kwargs)
        trial.compute_all_features()
        all_epochs.append(trial)

    for i, _ in enumerate(anx_epochs):
        trial = Trial(TrialLabel.CONTROL, anx_epochs[i], **trial_kwargs)
        trial.compute_all_features()
        all_epochs.append(trial)

    all_labels = [0] * len(normal_epochs) + [1] * len(anx_epochs)

    return all_epochs, all_labels


if __name__ == '__main__':
    anx_epochs = DaspsPreprocessor.get_severe_moderate_epochs()
    n_epochs = anx_epochs.get_data().shape[0]

    # conn = np.empty((n_epochs, n_channels, n_channels, n_bands))

    for i in range(n_epochs):
        print(i)

        trial = Trial(TrialLabel.CONTROL, anx_epochs[i])
        trial.compute_time_measures()

        # conn[i] = trial.compute_connectivity()
        # trial.compute_rel_powers()

        print(*[(key, trial.features[key])
              for key in trial.features], sep='\n')

        print("Total features: ", len(trial.features))

        print(trial.epoch.get_data().shape)

# spectral entropy, sample entropy, approximate entropy, lyapunov exponent, hurst exponent, higuchi fd

# All features:
# Freq: PSD, absolute power, relative power, mean power, asymmetry, band power ratios, HHT
# Time: hjorth, katz fd, higuchi fd, entropy, (lyapunov exponent), (detrended fluctuation analysis), hurst exp
# Time-frequency: DWT, RMS, energy
# Connectivity: PLI, wPLI, graph theory measures


# Classifiers: SVM, RF, Bagging, LightGBM, XGBoost, CatBoost, KNN, LDA, AdaBoost, Gradient Bagging, DT, MLP, SSAE, CNN, DBN, RBFNN, CNN, LSTM, NB
