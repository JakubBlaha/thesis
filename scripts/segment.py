import glob
import mne
import numpy as np
import pandas as pd
import csv
import os

from constants import TARGET_SAMPLING_FREQ as FS

script_dirname = os.path.dirname(os.path.abspath(__file__))

DASPS_CSV_PATH = os.path.abspath(
    os.path.join(script_dirname, "../data/DASPS.csv"))


def parse_dasps_situations() -> pd.DataFrame:
    with open(DASPS_CSV_PATH) as f:
        reader = csv.DictReader(f)

        d = {
            'subject_id': [],
            'sit_index': [],
            'valence': [],
            'arousal': []
        }

        for row in reader:
            _stripped = row['Id Participant'].strip('S')

            subject_id = int(_stripped) if _stripped else d["subject_id"][-1]
            sit_index = int(row['Id situation ']) - 1
            valence = int(row['valence'])
            arousal = int(row['Arousal'])

            d['subject_id'].append(subject_id)
            d['sit_index'].append(sit_index)
            d['valence'].append(valence)
            d['arousal'].append(arousal)

    return pd.DataFrame(d)


def get_sam_label(valence, arousal):
    if valence > 5 or arousal < 5:
        return 0
    elif valence in [0, 1, 2] and arousal in [7, 8, 9]:
        return 3
    elif valence in [2, 3, 4] and arousal in [6]:
        return 2
    elif valence in [4, 5] and arousal in [5]:
        return 1

    return 0


def add_sam_labels(epochs, subject_id, seconds_per_epoch):
    global situations

    samples_per_situation = FS * 30
    samples_per_epoch = FS * seconds_per_epoch

    labels = []

    for i in range(len(epochs)):
        epochs_starts_at_sample = samples_per_epoch * i
        situation_idx = epochs_starts_at_sample // samples_per_situation

        row = situations[(situations.subject_id == subject_id) &
                         (situations.sit_index == situation_idx)]
        assert len(row) == 1

        labels.append(
            get_sam_label(
                row['valence'].values[0],
                row['arousal'].values[0]))

    epochs.metadata = epochs.metadata.assign(sam=labels)


def segment_dasps(path, seconds_per_epoch, overlap, res_dir):
    orig_epochs = mne.read_epochs(path, preload=True)
    meta: pd.DataFrame = orig_epochs.metadata  # type: ignore

    data = orig_epochs.get_data()
    data = np.concatenate(data, axis=1)

    raw = mne.io.RawArray(data, orig_epochs.info)

    # Overlap is disabled!
    epochs = mne.make_fixed_length_epochs(
        raw, duration=seconds_per_epoch, overlap=overlap * seconds_per_epoch *
        0, preload=True)

    # Copy original recording metadata to each epoch
    epochs.metadata = pd.DataFrame(
        {k: [v[0]] * len(epochs) for k, v in meta.items()})

    dataset = meta["dataset"][0]
    subject_id = meta['subject'][0]

    # Add SAM labels to metadata
    if dataset == "dasps":
        add_sam_labels(epochs, subject_id, seconds_per_epoch)

    path = os.path.join(res_dir, f"S{subject_id:03d}-epo.fif")
    epochs.save(path, overwrite=True)


def segment_sad(path, seconds_per_epoch, overlap, res_dir):
    """
    Segment SAD (Social Anxiety Disorder) dataset recordings into fixed-length epochs.

    Args:
        path (str): Path to the SAD dataset recording file.
        seconds_per_epoch (int): Duration of each epoch in seconds.
        overlap (float): Overlap between consecutive epochs as a fraction.
        res_dir (str): Directory to save the segmented epochs.

    Returns:
        None: The function saves the epoched data to a file.
    """
    fif = mne.io.read_raw(path, preload=True)

    # # Print recording length
    # print(f"Recording length: {fif.times[-1]}s")

    # Segment into epochs
    epochs = mne.make_fixed_length_epochs(
        fif, duration=seconds_per_epoch, overlap=overlap * seconds_per_epoch *
        0, preload=True)

    # Severity for SAD is represented by the first digit in the filename, lowest severity starts at 1
    # therefore we substract one to make it consistent with DASPS
    severity = int(os.path.basename(path).split('.')[0][1]) - 1
    subject_id = int(os.path.basename(path).split('.')[0][1:])

    # Set metadata for each epoch
    metadata = {
        'dataset': ['SAD'] * len(epochs),
        'stai': [severity] * len(epochs),
        'subject': [subject_id] * len(epochs)
    }
    epochs.metadata = pd.DataFrame(metadata)

    # Save
    new_path = res_dir + f"S{subject_id:03d}-epo.fif"
    epochs.save(new_path, overwrite=True)


def validate_seglen(seglen):
    """
    Validate if the segment length is appropriate for the dataset.

    Args:
        seglen (int): Segment length in seconds.

    Raises:
        AssertionError: If segment length is invalid (not greater than 0, 
                        greater than 30, or not a divisor of 30).

    Returns:
        None
    """
    assert seglen > 0, "Segment length must be greater than 0"

    assert seglen <= 30, "Segment length must be less than or equal to 30"

    assert 30 % seglen == 0, (
        "Cannot segment into epochs that would match the DASPS situation length. " +
        "Please change SECONDS_PER_EPOCH to a divisor of 30")


def segment(seconds_per_epoch: int):
    """
    Main function to segment EEG recordings into fixed-length epochs.

    This function processes all matching files and segments them based on their dataset type
    (DASPS or SAD) using the appropriate segmentation function.

    Args:
        seconds_per_epoch (int): Duration of each epoch in seconds.

    Returns:
        None: Files are saved to the result directory.
    """
    res_dir = os.path.join(
        script_dirname, f"../data/segmented/{seconds_per_epoch}s/raw/")
    os.makedirs(res_dir, exist_ok=True)

    fif_pattern = os.path.abspath(os.path.join(
        script_dirname, "../data/fif/S???.fif"))
    paths = glob.glob(fif_pattern)

    print(script_dirname)
    print(fif_pattern)

    validate_seglen(seconds_per_epoch)

    print(paths)

    for path in paths:
        basename = os.path.basename(path)
        s_number = basename.split('.')[0][1:]

        args = [path, seconds_per_epoch, 0, res_dir]

        if s_number.startswith('0'):
            segment_dasps(*args)
        else:
            segment_sad(*args)


situations = parse_dasps_situations()

if __name__ == "__main__":
    segment(10)
