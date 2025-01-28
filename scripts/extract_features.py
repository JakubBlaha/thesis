# %%
import glob
import os
import mne
import mne_features
import mne_connectivity
import numpy as np
import pandas as pd
import logging

from constants import CHANNEL_NAMES, TARGET_SAMPLING_FREQ as sfreq, SAD_MULTIPLY_FACTOR

logger = logging.getLogger(__name__)

n_channels = len(CHANNEL_NAMES)
freq_bands = {
    # 'delta': (0.5, 4), # delta has been filtered out
    'theta': [4, 8],
    'alpha-1': [8, 10],
    'alpha-2': [10, 13],
    'beta': [13, 30],
    'gamma': [30, 45],  # above 45 has been filtered out
}
freq_band_names = list(freq_bands.keys())
min_freqs = [freq_bands[band][0] for band in freq_bands]
max_freqs = [freq_bands[band][1] for band in freq_bands]
n_bands = len(freq_bands)

# Left channels end with odd numbers, right channels end with even numbers
left_channels = sorted([i for i in CHANNEL_NAMES if int(i[-1]) % 2 == 1])
right_channels = sorted([i for i in CHANNEL_NAMES if int(i[-1]) % 2 == 0])

out_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../data/features"))

os.makedirs(out_dir, exist_ok=True)

segmented_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/segmented"))


def get_epoch_features(epoch):
    d = {}

    epoch = np.expand_dims(epoch, axis=0)

    # Time complexity features
    logging.info("Extracting time complexity features")

    funcs = [
        'hjorth_mobility', 'hjorth_complexity', 'variance', 'app_entropy',
        'line_length', 'skewness', 'kurtosis', 'rms', 'decorr_time',
        'higuchi_fd', 'katz_fd', 'samp_entropy', 'hurst_exp']

    x = mne_features.feature_extraction.extract_features(
        epoch, sfreq, funcs, n_jobs=1)
    x = x.reshape((len(funcs), n_channels))

    for func_name, channels in zip(funcs, x):
        for ch_val, ch_name in zip(channels, CHANNEL_NAMES):
            feat_name = f'time_{func_name}_{ch_name}'
            d[feat_name] = ch_val

    # Band power features
    logging.info("Extracting band power features")

    for normalize in [False, True]:
        freq_bands_ = np.asanyarray(
            [freq_bands[band][0] for band in freq_bands] +
            [freq_bands['gamma'][1]])

        powers_and_ratios = mne_features.univariate.compute_pow_freq_bands(
            sfreq, epoch[0],
            freq_bands_, normalize=normalize, ratios='all', psd_method='welch',
            psd_params={'welch_n_overlap': sfreq // 2})
        num_powers = n_channels * n_bands

        # Powers and ratios are returned in a single array, need to split them

        # Shape is (n_channels, n_bands)
        powers = powers_and_ratios[:num_powers].reshape((n_channels, -1))
        # print(powers.shape)

        # Shape is (n_channels, n_bands, n_bands-1)
        ratios = powers_and_ratios[num_powers:]
        pow_ratios = ratios.reshape(n_channels, n_bands, -1)

        # TODO also save power ratios

        for el_idx, el in enumerate(powers):
            for band_idx, band_pow in enumerate(el):
                feat_name = f'{
                    'rel' if normalize else 'abs'}_pow_{
                    freq_band_names[band_idx]}_{
                    CHANNEL_NAMES[el_idx]}'
                d[feat_name] = band_pow

    # Connectivity features
    logging.info("Extracting connectivity features")

    min_freq = min(min_freqs)
    max_freq = max(max_freqs)

    freqs = np.linspace(min_freq, max_freq, int(
        (max_freq - min_freq) * 4 + 1))

    res = mne_connectivity.spectral_connectivity_time(
        epoch, freqs=freqs, method="wpli", sfreq=sfreq, mode="cwt_morlet",
        fmin=min_freqs, fmax=max_freqs, faverage=True, n_jobs=1, verbose=0).get_data()

    conn_of_one_epoch = res[0]
    matrix = conn_of_one_epoch.reshape(
        (n_channels, n_channels, n_bands))

    matrix = np.moveaxis(matrix, 2, 0)

    for band_idx, band in enumerate(matrix):
        for el_idx, el in enumerate(band):
            for el2_idx, el2 in enumerate(el[:el_idx]):
                feat_name = f'conn_{freq_band_names[band_idx]}_{
                    CHANNEL_NAMES[el_idx]}_{CHANNEL_NAMES[el2_idx]}'
                d[feat_name] = el2

    # Asymmetry index features
    logging.info("Extracting asymmetry index features")

    for band_name in freq_band_names:
        for left_ch_name, right_ch_name in zip(left_channels, right_channels):
            left_abs_pow = d['abs_pow_' + band_name + '_' + left_ch_name]
            right_abs_pow = d['abs_pow_' + band_name + '_' + right_ch_name]

            asym_idx = (
                left_abs_pow - right_abs_pow) / (left_abs_pow + right_abs_pow)

            d[f'ai_{band_name}_{right_ch_name}-{left_ch_name}'] = asym_idx

    return d


def get_features_of_all_epochs(path) -> pd.DataFrame:
    print(path)

    epochs = mne.read_epochs(path)
    meta: pd.DataFrame = epochs.metadata  # type: ignore

    # Retrieve metadata about recording
    subject_id = meta.iloc[0]['subject']
    dataset = meta.iloc[0]['dataset']

    # HAM label is the same for all epochs in the DASPS dataset
    ham_label = None

    if "ham" in meta.columns:
        ham_label = meta.iloc[0]["ham"]

    # STAI score is the same for all epochs in the SAD dataset
    sad_severity = None

    if "stai" in meta.columns:
        sad_severity = meta.iloc[0]["stai"]

    # Build a dict of features for each epoch
    epoch_dicts = []

    for index, epoch in enumerate(epochs):
        # SAM label is different for each epoch
        sam_label = None

        if "sam" in meta.columns:
            sam_label = meta.iloc[index]["sam"]

        if dataset == "SAD":
            epoch = (epoch.T * SAD_MULTIPLY_FACTOR).T  # type: ignore

        d = get_epoch_features(epoch)

        d.update({
            'subject': subject_id,
            'dataset': dataset,
            'uniq_subject_id': f'{dataset}_{subject_id}',
            'sam': sam_label,
            'ham': ham_label,
            'stai': sad_severity
        })

        epoch_dicts.append(d)

    return pd.DataFrame(epoch_dicts)


def extract_features_from_all_segments():
    pattern = os.path.join(segmented_path, "*/clean/*-epo.fif")
    paths_to_subject_epochs = glob.glob(pattern)

    # Split paths into a dict by segment lengths
    paths_by_segment_length = {}

    for path in paths_to_subject_epochs:
        segment_length = int(os.path.basename(
            os.path.dirname(os.path.dirname(path))).strip('s'))

        if segment_length not in paths_by_segment_length:
            paths_by_segment_length[segment_length] = []

        paths_by_segment_length[segment_length].append(path)

    for seglen, paths in paths_by_segment_length.items():
        output_file_path = f'{out_dir}/features_{seglen}s.csv'

        if os.path.exists(output_file_path):
            print(f"Skipping existing {output_file_path}")
            continue

        print(f"Extracting features for {seglen}s segments")

        all_dfs = [get_features_of_all_epochs(path) for path in paths]

        df = pd.concat(all_dfs)

        # Save features into CSV for segment length
        df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    extract_features_from_all_segments()

# %%
