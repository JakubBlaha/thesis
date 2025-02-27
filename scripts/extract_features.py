# %%
import glob
import os
import mne
import mne_features
import mne_connectivity
import numpy as np
import pandas as pd
import logging
import concurrent.futures
from functools import partial
import multiprocessing
from datetime import datetime

from constants import CHANNEL_NAMES, TARGET_SAMPLING_FREQ as sfreq, SAD_MULTIPLY_FACTOR

# Setup logging to file
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Get the number of CPU cores available
N_CORES = multiprocessing.cpu_count()

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
    logger.info("Extracting time complexity features")

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
    logger.info("Extracting band power features")

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
    logger.info("Extracting connectivity features")

    min_freq = min(min_freqs)
    max_freq = max(max_freqs)

    freqs = np.linspace(min_freq, max_freq, int(
        (max_freq - min_freq) * 4 + 1))

    # Frequency resolution is 1 Hz
    res = mne_connectivity.spectral_connectivity_time(
        epoch, freqs=freqs, method="wpli", sfreq=sfreq, mode="multitaper", n_cycles=4,
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
    logger.info("Extracting asymmetry index features")

    for band_name in freq_band_names:
        for left_ch_name, right_ch_name in zip(left_channels, right_channels):
            left_abs_pow = d['abs_pow_' + band_name + '_' + left_ch_name]
            right_abs_pow = d['abs_pow_' + band_name + '_' + right_ch_name]

            asym_idx = (
                left_abs_pow - right_abs_pow) / (left_abs_pow + right_abs_pow)

            d[f'ai_{band_name}_{right_ch_name}-{left_ch_name}'] = asym_idx

    return d


def process_single_epoch(epoch_index, epoch, meta, dataset, subject_id, ham_label, sad_severity):
    """Process a single epoch and return its features dict"""
    try:
        # SAM label is different for each epoch
        sam_label = None

        if "sam" in meta.columns:
            sam_label = meta.iloc[epoch_index]["sam"]

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

        return d
    except Exception as e:
        logger.error(f"Error processing epoch {epoch_index} for {dataset}_{subject_id}: {str(e)}", exc_info=True)
        return None


def get_features_of_all_epochs(path, max_workers=None) -> pd.DataFrame:
    """Extract features from all epochs in a file with parallel processing"""
    logger.info(f"Processing {path}")

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

    if max_workers is None:
        max_workers = max(1, N_CORES - 1)  # Leave one core free
    
    process_func = partial(
        process_single_epoch,
        meta=meta,
        dataset=dataset, 
        subject_id=subject_id,
        ham_label=ham_label,
        sad_severity=sad_severity
    )
    
    # Build a dict of features for each epoch
    epoch_dicts = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, i, epoch) for i, epoch in enumerate(epochs)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    epoch_dicts.append(result)
            except Exception as e:
                logger.error(f"Error in future for {path}: {str(e)}", exc_info=True)
    
    return pd.DataFrame(epoch_dicts)


def process_subject_file(path, seglen):
    """Process a single subject file and return its DataFrame"""
    try:
        return get_features_of_all_epochs(path)
    except Exception as e:
        subject_id = os.path.basename(path).split('-')[0]
        logger.error(f"Error processing {path} for subject {subject_id}: {str(e)}", exc_info=True)
        return pd.DataFrame()


def extract_features_from_all_segments(seglen=None, max_workers=None):
    """
    Extract features from segments with parallel processing
    
    Parameters:
    -----------
    seglen : int, optional
        If provided, only extract features for segments of this length (in seconds)
    max_workers : int, optional
        Number of worker processes to use
    """
    pattern = os.path.join(segmented_path, "*/clean/*-epo.fif")
    paths_to_subject_epochs = glob.glob(pattern)
    
    logger.info(f"Found {len(paths_to_subject_epochs)} files to process")

    # Split paths into a dict by segment lengths
    paths_by_segment_length = {}

    for path in paths_to_subject_epochs:
        segment_length = int(os.path.basename(
            os.path.dirname(os.path.dirname(path))).strip('s'))

        if segment_length not in paths_by_segment_length:
            paths_by_segment_length[segment_length] = []

        paths_by_segment_length[segment_length].append(path)

    if max_workers is None:
        max_workers = N_CORES
    
    # If seglen is specified, only process that segment length
    if seglen is not None:
        if seglen in paths_by_segment_length:
            paths_by_segment_length = {seglen: paths_by_segment_length[seglen]}
        else:
            logger.warning(f"No segments found with length {seglen}s")
            return
    
    for seglen, paths in paths_by_segment_length.items():
        output_file_path = f'{out_dir}/features_{seglen}s.csv'

        if os.path.exists(output_file_path):
            logger.info(f"Skipping existing {output_file_path}")
            continue

        logger.info(f"Extracting features for {seglen}s segments ({len(paths)} files)")
        
        all_dfs = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(process_subject_file, seglen=seglen)
            futures = [executor.submit(process_func, path) for path in paths]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    df = future.result()
                    if not df.empty:
                        all_dfs.append(df)
                    logger.info(f"Completed {i+1}/{len(paths)} files for {seglen}s segments")
                except Exception as e:
                    logger.error(f"Error in future for {seglen}s segment: {str(e)}", exc_info=True)

        if all_dfs:
            df = pd.concat(all_dfs)
            df.to_csv(output_file_path, index=False)
            logger.info(f"Saved features for {seglen}s segments to {output_file_path}")
        else:
            logger.warning(f"No data processed for {seglen}s segments")


if __name__ == "__main__":
    logger.info("Starting feature extraction process")
    try:
        extract_features_from_all_segments()
        logger.info("Feature extraction completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in main process: {str(e)}", exc_info=True)
        raise

# %%
