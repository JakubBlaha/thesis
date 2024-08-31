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
    def __init__(self, trial_label: TrialLabel, epoch: mne.EpochsArray) -> None:
        self.trial_label = trial_label
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

        funcs = ['hjorth_mobility', 'hjorth_complexity', 'variance',
                 'app_entropy', 'samp_entropy', 'higuchi_fd', 'katz_fd']
        # extractor = feature_extraction.FeatureExtractor(sfreq, funcs)
        data = self.epoch.get_data()[0]

        for func_name in funcs:
            print(func_name)
            vec = mne_features.get_univariate_funcs(sfreq)[
                func_name](data)

            for ch_val, ch_name in zip(vec, CHANNEL_NAMES):
                feat_name = f'{func_name}_all_{ch_name}'
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
            self.epoch, freqs=freqs, method="wpli", sfreq=sfreq, mode="cwt_morlet", fmin=min_freqs, fmax=max_freqs,
            faverage=True).get_data()

        conn_of_one_epoch = res[0]
        matrix = conn_of_one_epoch.reshape(
            (n_channels, n_channels, n_bands))

        matrix = np.moveaxis(matrix, 2, 0)

        for band_idx, band in enumerate(matrix):
            for el_idx, el in enumerate(band):
                for el2_idx, el2 in enumerate(el[:el_idx]):
                    feat_name = f'wpli_{freq_band_names[band_idx]}_{
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

    # def plot_beta(self):
    #     powers = [el.rel_pow_alpha_2 for el in self.electrodes.values()]

    #     print(self.info)

    #     mne.viz.topomap.plot_topomap(
    #         powers, self.info, show=True, names=CHANNEL_NAMES)

    # def get_asym_vect(self):


if __name__ == '__main__':
    anx_epochs = DaspsPreprocessor.get_severe_moderate_epochs()
    n_epochs = anx_epochs.get_data().shape[0]

    conn = np.empty((n_epochs, n_channels, n_channels, n_bands))

    for i in range(n_epochs):
        print(i)
        trial = Trial(TrialLabel.CONTROL, anx_epochs[i])

        # trial.show_spectrogram()

        trial.compute_time_measures()
        trial.compute_powers(normalize=True)
        trial.compute_powers(normalize=False)
        trial.compute_connectivity()
        trial.compute_asymmetry()

        # conn[i] = trial.compute_connectivity()
        # trial.compute_rel_powers()

        print(*[(key, trial.features[key])
              for key in trial.features], sep='\n')

        print("Total features: ", len(trial.features))

        break

# spectral entropy, sample entropy, approximate entropy, lyapunov exponent, hurst exponent, higuchi fd

# All features:
# Freq: PSD, absolute power, relative power, mean power, asymmetry, band power ratios, HHT
# Time: hjorth, katz fd, higuchi fd, entropy, (lyapunov exponent), (detrended fluctuation analysis), hurst exp
# Time-frequency: DWT, RMS, energy
# Connectivity: PLI, wPLI, graph theory measures
