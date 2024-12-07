# %%
from autoreject import AutoReject
import os
import glob
import mne


def autoreject_file(path):
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
    script_dirname = os.path.dirname(os.path.abspath(__file__))
    fnames = glob.glob(os.path.join(script_dirname,
                                    "../data/segmented/*/raw/*-epo.fif"))

    for fname in fnames:
        autoreject_file(fname)


if __name__ == "__main__":
    run_autoreject()

# %%
