"""
AutoReject Runner Script

This script applies AutoReject to preloaded epochs from EEG/MEG recordings to
automatically clean artifacts. It processes all epoch files (*.fif) in the specified
directory structure and saves the cleaned epochs to a parallel 'clean' directory.
"""

from autoreject import AutoReject
import os
import glob
import mne


def autoreject_file(path):
    """
    Process a single epochs file with AutoReject.

    For each file, this function:
    1. Determines the output location in a parallel 'clean' directory
    2. Skips processing if the output file already exists
    3. Loads the epochs data
    4. Applies AutoReject to clean artifacts
    5. Saves the cleaned epochs to the output location

    Parameters
    ----------
    path : str
        Path to the epochs file (.fif) to be processed

    Returns
    -------
    None
    """
    # Skip existing files to avoid unnecessary computation
    out_dir = os.path.join(os.path.dirname(os.path.dirname(path)), 'clean')
    basename = os.path.basename(path)
    new_path = os.path.join(out_dir, basename)

    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(new_path):
        print("Skipping existing", path)
        return

    # Load data
    epochs = mne.read_epochs(path, preload=True)
    num_epochs = len(epochs)

    # Apply autoreject
    ar = AutoReject(random_state=87, cv=num_epochs)
    epochs_clean = ar.fit_transform(epochs)

    epochs_clean.save(new_path, overwrite=True)


def run_autoreject():
    """
    Run AutoReject on all epoch files in the data directory.

    This function:
    1. Finds all epoch files (ending with -epo.fif) in the data/segmented/*/raw/ directories
    2. Processes each file with AutoReject using the autoreject_file function

    Returns
    -------
    None
    """
    script_dirname = os.path.dirname(os.path.abspath(__file__))
    fnames = glob.glob(os.path.join(script_dirname,
                                    "../data/segmented/*/raw/*-epo.fif"))

    for fname in fnames:
        autoreject_file(fname)


if __name__ == "__main__":
    run_autoreject()
