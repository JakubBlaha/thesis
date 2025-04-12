"""
This script generates a plot visualizing EEG signals in individual frequency bands.
The individual bands (Delta, Theta, Alpha, Beta, Gamma) are obtained by filtering the original signal.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Visualize EEG frequency bands.')
    parser.add_argument('fif_file', type=str,
                        help='Path to the .fif file containing epochs')
    args = parser.parse_args()

    # Use the provided fif file path
    fif_file = args.fif_file

    # Load the epochs data
    epochs = mne.read_epochs(fif_file, preload=True)

    # Pick only EEG channels (ignore e.g. EOG, ECG, etc.)
    epochs.pick_types(eeg=True)

    # Define frequency bands (common definitions)
    # Adjust the exact cutoffs as needed for your application
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }

    # Get the sampling frequency
    sfreq = epochs.info['sfreq']

    # Get data from the first epoch and first channel
    # first epoch, first channel, all time points
    data_full = epochs.get_data()[0, 0, :]
    times_full = epochs.times  # Get time points from epochs

    # Limit to x timepoints (or all if fewer)
    n_points = min(129, len(times_full))
    data = data_full[:n_points]
    times = times_full[:n_points]

    # Set font to serif for the entire plot
    plt.rcParams['font.family'] = 'serif'

    # Create subplots: one row per band
    fig, axes = plt.subplots(len(bands), 1, figsize=(5, 4), sharex=True)

    # Ensure axes is a list (in case of only one band, it becomes a single object)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # First compute all filtered data to find global min/max
    filtered_data = []
    for band_name, (l_freq, h_freq) in bands.items():
        # Use MNE's built-in filter_data function to band-pass filter
        filtered = mne.filter.filter_data(
            data, sfreq, l_freq, h_freq, verbose=False)
        filtered_data.append(filtered)

    # Find global min and max for consistent y-axis scaling
    global_min = min(np.min(f) for f in filtered_data)
    global_max = max(np.max(f) for f in filtered_data)

    # Add a small margin (5%) to min/max for better visualization
    y_margin = 0.05 * (global_max - global_min)
    y_min = global_min - y_margin
    y_max = global_max + y_margin

    # Iterate through each band and plot with same scale
    for idx, ((band_name, (l_freq, h_freq)),
              filtered) in enumerate(zip(bands.items(),
                                         filtered_data)):
        axes[idx].plot(times, filtered, color='b', lw=1)
        axes[idx].set_ylabel(band_name)
        axes[idx].set_xlim([times[0], times[-1]])
        # Set same y-axis limits for all plots
        axes[idx].set_ylim([y_min, y_max])
        # Hide y-axis tick labels
        axes[idx].set_yticklabels([])

    # Label the x-axis of the bottom subplot
    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
