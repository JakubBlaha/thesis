from enum import Enum

import mne
import mne_features
import mne_connectivity
import numpy as np
import io
from mne_features import univariate
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
from PIL import Image

CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7',
                 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

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
        funcs = ['hjorth_mobility', 'hjorth_complexity', 'variance', 'app_entropy',
                 'line_length', 'skewness', 'kurtosis', 'rms', 'decorr_time', 'higuchi_fd']

        # More resource demanding features
        # funcs += ['higuchi_fd', 'katz_fd', 'samp_entropy', 'hurst_exp']

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
        freq_bands_ = np.asanyarray([freq_bands[band][0]
                                     for band in freq_bands] + [freq_bands['gamma'][1]])

        data = self.epoch.get_data()[0]

        powers_and_ratios = univariate.compute_pow_freq_bands(
            sfreq, data,
            freq_bands_, normalize=normalize, ratios='all', psd_method='welch',
            psd_params={'welch_n_overlap': sfreq // 2})
        num_powers = n_channels * n_bands
        powers = powers_and_ratios[:num_powers].reshape((n_channels, -1))

        # shape is (n_channels, n_bands, n_bands-1)
        ratios = powers_and_ratios[num_powers:]
        self.pow_ratios = ratios.reshape(n_channels, n_bands, -1)

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

    def compute_power_matrices(self):
        freq_bands_ = np.asanyarray([freq_bands[band][0]
                                     for band in freq_bands] + [freq_bands['gamma'][1]])

        powers = mne_features.univariate.compute_pow_freq_bands(
            sfreq, self.epoch.get_data()[0],
            freq_bands_, normalize=True, psd_method='welch',
            psd_params={'welch_n_overlap': sfreq // 2})
        powers = powers[:n_channels *
                        n_bands].reshape((n_channels, -1))
        powers = powers.swapaxes(0, 1)

        matrices = []

        for i in range(n_bands):
            fig, ax = plt.subplots(frameon=False)
            im, _ = mne.viz.plot_topomap(powers[i], self.epoch.info, contours=0, sensors=False,
                                         outlines=None, res=32, cmap="Greys", axes=ax, show=False)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)

            image = Image.open(buf)
            image = image.resize((32, 32))
            gray_image = image.convert('L')
            pixel_array = np.asarray(gray_image)

            matrices.append(pixel_array)

        self.power_matrices = np.array(matrices)

    def compute_all_features(self):
        self.compute_time_measures()
        self.compute_powers(normalize=True)
        self.compute_powers(normalize=False)
        self.compute_connectivity()
        self.compute_asymmetry()

    def select_features(self):
        # Eliminate redundant features by variance
        features = np.array([list(self.features.values())])
        selector = VarianceThreshold(threshold=0.01)
        reduced = selector.fit_transform(features)
        features_kept = features.columns[selector.get_support(indices=True)]

        print(features_kept)

        return reduced
