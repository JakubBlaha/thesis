"""
EEG Dataset Conversion Script.

This script converts EEG data from DASPS and SAD datasets to the FIF format used by MNE.
The script handles different preprocessing steps including filtering, resampling,
channel selection, and re-referencing to ensure consistency between datasets.
"""

import mne
import os
import h5py
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

from constants import CHANNEL_NAMES, TARGET_SAMPLING_FREQ

# Path to the preprocessed DASPS dataset
DASPS_PREP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../data/datasets/DASPS"))

# Path to the preprocessed SAD dataset
SAD_PREP_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../data/datasets/SAD/preprocessed"))

# Mapping of severity labels to numerical values
severity_to_number = {
    "control": 1,
    "mild": 2,
    "moderate": 3,
    "severe": 4
}

# Output directory for storing converted FIF files
FIF_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../data/fif"))

# -------------- DASPS specific ---------------
# Mapping of HAM depression severity scores to subject IDs in the DASPS dataset
# 0: No depression, 1: Mild, 2: Moderate, 3: Severe
DASPS_HAM_SEVERITY_SUBJECTS_MAP = {
    0: [10, 14],
    1: [8, 9, 15, 18, 23],
    2: [11, 13, 21],
    3: [1, 2, 3, 4, 5, 6, 7, 12, 16, 17, 19, 20, 22]
}

# Parameters for bandpass filtering (4-40 Hz)
BANDPASS_FILTER_KW = {
    "l_freq": 4,  # Lower frequency bound (Hz)
    "h_freq": 40,  # Upper frequency bound (Hz)
    "h_trans_bandwidth": 2,  # Transition bandwidth for upper bound (Hz)
}

# Parameters for notch filtering around 50 Hz (power line noise)
BANDSTOP_FILTER_KW = {
    "l_freq": 52,  # Upper frequency bound (Hz)
    "h_freq": 48,  # Lower frequency bound (Hz)
    "h_trans_bandwidth": 1,  # Transition bandwidth for upper bound (Hz)
    "l_trans_bandwidth": 1   # Transition bandwidth for lower bound (Hz)
}


def get_epochs_from_mat(fname, subject_id):
    """
    Convert a .mat file from the DASPS dataset into MNE Epochs.

    Parameters:
    - fname: str, filename of the .mat file
    - subject_id: int, ID of the subject

    Returns:
    - epochs: MNE Epochs object
    """
    fname = os.path.join(DASPS_PREP_PATH, fname)

    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])

    info = mne.create_info(
        ch_names=CHANNEL_NAMES, ch_types="eeg", sfreq=TARGET_SAMPLING_FREQ)
    info.set_montage('standard_1020')

    num_epochs = data.shape[0]

    # Concatenate epochs
    data = data.reshape(-1, 14)

    # Make raw object
    raw = mne.io.RawArray(data.T, info)

    # Filtering
    raw.filter(**BANDPASS_FILTER_KW)  # type: ignore
    raw.filter(**BANDSTOP_FILTER_KW)  # type: ignore

    epochs = mne.make_fixed_length_epochs(raw, 15, preload=True)

    # Add metadata
    epochs.metadata = pd.DataFrame(
        {"subject": [subject_id] * num_epochs,
         "dataset": ["dasps"] * num_epochs})

    # Re-reference to average
    epochs.set_eeg_reference("average", projection=False)

    return epochs


def _ensure_fif_dir():
    """
    Ensure the output directory for FIF files exists.
    """
    os.makedirs(FIF_DATA_PATH, exist_ok=True)


def convert_dasps_to_fif():
    """
    Convert DASPS dataset to FIF format.

    This function processes all .mat files in the DASPS dataset directory,
    applies preprocessing steps, and saves the data in FIF format.
    """
    _ensure_fif_dir()

    mat_paths = glob.glob(DASPS_PREP_PATH + "/*.mat")

    for path in mat_paths:
        fname = os.path.basename(path)
        subject_id = int(fname.strip("preprocessed.mat").strip("S"))

        # Get epochs
        epochs = get_epochs_from_mat(fname, subject_id)

        # Add HAM labeling
        for severity, subjects in DASPS_HAM_SEVERITY_SUBJECTS_MAP.items():
            if subject_id in subjects:
                epochs.metadata["ham"] = [severity] * len(epochs)
                break

        fname = "S" + str(subject_id).zfill(3) + ".fif"
        fpath = os.path.join(FIF_DATA_PATH, fname)

        epochs.save(fpath, overwrite=True)


# -------------- SAD specific ---------------
def convert_sad_to_fif():
    """
    Convert SAD dataset to FIF format.

    This function processes all .edf files in the SAD dataset directory,
    applies preprocessing steps, and saves the data in FIF format.
    """
    _ensure_fif_dir()

    edf_paths = sorted(glob.glob(SAD_PREP_PATH + "/*/*.edf"))

    for index, path in enumerate(edf_paths):
        # Data are already re-referenced to average reference
        raw = mne.io.read_raw_edf(path, preload=True)

        # Rename channels to match DASPS dataset
        raw.rename_channels({
            'Fp1': 'AF3',
            'Fp2': 'AF4',
        })

        # Leave only channels common with DASPS dataset
        raw = raw.pick_channels(CHANNEL_NAMES)

        raw.set_montage("standard_1020")

        # Filtering
        raw.filter(**BANDPASS_FILTER_KW)  # type: ignore
        raw.filter(**BANDSTOP_FILTER_KW)  # type: ignore

        # Downsample to 128 Hz
        raw = raw.resample(128)

        # Re-reference to average reference after dropping channels
        raw.set_eeg_reference("average", projection=False)

        # Rescale the data to the same scale as DASPS
        raw._data *= 1e6

        # Get subject number from basename
        s_number = os.path.basename(path).split(".")[0].replace("C", "")

        # Get severity
        s_severity = os.path.basename(os.path.dirname(path))
        s_severity_as_number = severity_to_number[s_severity]

        # Save as ScXX.fif, denoting the SAD dataset, where
        # c 1=CONTROL, 2=MILD, 3=MODERATE, 4=SEVERE
        fname = "S" + str(s_severity_as_number) + s_number.zfill(2) + ".fif"
        new_path = os.path.join(FIF_DATA_PATH, fname)

        raw.save(new_path, overwrite=True)


if __name__ == "__main__":
    convert_dasps_to_fif()
    convert_sad_to_fif()
