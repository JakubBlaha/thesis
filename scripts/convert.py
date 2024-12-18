# %%
import mne
import os
import h5py
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

from .constants import CHANNEL_NAMES, TARGET_SAMPLING_FREQ

DASPS_PREP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../data/datasets/DASPS"))

SAD_PREP_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../data/datasets/SAD/preprocessed"))

severity_to_number = {
    "control": 1,
    "mild": 2,
    "moderate": 3,
    "severe": 4
}

FIF_DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../data/fif"))

# -------------- DASPS specific ---------------
DASPS_HAM_SEVERITY_SUBJECTS_MAP = {
    0: [10, 14],
    1: [8, 9, 15, 18, 23],
    2: [11, 13, 21],
    3: [1, 2, 3, 4, 5, 6, 7, 12, 16, 17, 19, 20, 22]
}


def get_epochs_from_mat(fname, subject_id):
    fname = os.path.join(DASPS_PREP_PATH, fname)

    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])

    info = mne.create_info(
        ch_names=CHANNEL_NAMES, ch_types="eeg", sfreq=TARGET_SAMPLING_FREQ)
    info.set_montage('standard_1020')

    data = data.swapaxes(1, 2)
    epochs = mne.EpochsArray(data, info)

    # Add metadata
    epochs.metadata = pd.DataFrame(
        {"subject": [subject_id] * len(epochs),
         "dataset": ["dasps"] * len(epochs)})

    # epochs.compute_psd(fmin=0, fmax=64).plot()
    # plt.show()

    # Filtering
    epochs.filter(l_freq=4, h_freq=30)
    epochs.filter(l_freq=52, h_freq=48,
                  l_trans_bandwidth=1, h_trans_bandwidth=1)

    # epochs.compute_psd(fmin=0, fmax=64).plot(dB=False)
    # plt.savefig("./figures/DASPS/psd/" + str(subject_id) + ".png", dpi=300)

    # Re-reference to average
    epochs.set_eeg_reference("average", projection=False)

    return epochs


def _ensure_fif_dir():
    os.makedirs(FIF_DATA_PATH, exist_ok=True)


def convert_dasps_to_fif():
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

        # raw.compute_psd(fmin=0, fmax=64).plot()
        # plt.show()

        # Apply a filter of 4 -- 30 Hz, as done in the DASPS preprocessed dataset
        raw.filter(l_freq=4, h_freq=30)
        raw.filter(l_freq=52, h_freq=48,
                   l_trans_bandwidth=1, h_trans_bandwidth=1)

        # Downsample to 128 Hz
        raw = raw.resample(128)

        # Re-reference to average reference after dropping channels
        raw.set_eeg_reference("average", projection=False)

        # Rescale the data to the same scale as DASPS
        raw._data *= 1e6

        # raw.compute_psd(fmin=0, fmax=64).plot(dB=False)
        # plt.savefig("./figures/SAD/psd/" + str(index) + ".png", dpi=300)

        # Get subject number
        s_number = path.split("/")[-1].split(".")[0].replace("C", "")

        # Get severity
        s_severity = path.split("/")[-2]
        s_severity_as_number = severity_to_number[s_severity]

        # Save as ScXX.fif, denoting the SAD dataset, where
        # c 1=CONTROL, 2=MILD, 3=MODERATE, 4=SEVERE
        fname = "S" + str(s_severity_as_number) + s_number.zfill(2) + ".fif"
        new_path = os.path.join(FIF_DATA_PATH, fname)

        raw.save(new_path, overwrite=True)


if __name__ == "__main__":
    convert_dasps_to_fif()
    convert_sad_to_fif()

# %%
