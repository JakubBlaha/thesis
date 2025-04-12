"""
This script creates topographical plots visualizing EEG features on a standard 10-20
electrode layout. It generates two types of plots:

1. Features by Domain: Shows which electrodes contribute to different feature domains
   (time domain, absolute power, relative power, asymmetry index) using color-coded
   half-circles overlaid on electrode positions.

2. Features by Frequency: Shows which electrodes contribute to different frequency bands
   (theta, alpha-1, alpha-2, beta, gamma) using color-coded half-circles.

These plots provide a spatial representation of which brain regions and frequency bands
are most relevant for classification according to the selected features.
"""

# %%
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import pandas as pd
from matplotlib import rcParams
import os

script_dir = os.path.dirname(__file__)
plots_dir = os.path.join(script_dir, '../data/plots')

# Set global font to serif
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']


def plot_features_by_domain(selected_features, channel_names, info):
    """Plot selected features grouped by domain (time, power, etc.)"""
    # Group features by domain
    time_features = [f for f in selected_features if f.startswith('time_')]
    abs_pow_features = [f for f in selected_features
                        if f.startswith('abs_pow_')]
    rel_pow_features = [f for f in selected_features
                        if f.startswith('rel_pow_')]
    ai_features = [f for f in selected_features if f.startswith('ai_')]

    # Define domain colors
    domain_colors = {
        'time': 'blue',
        'abs_pow': 'red',
        'rel_pow': 'green',
        'ai': 'orange'
    }

    # Define human-readable domain names for legend
    domain_labels = {
        'time': 'Time (Complexity)',
        'abs_pow': 'Abs. Power',
        'rel_pow': 'Rel. Power',
        'ai': 'Asym. Index'
    }

    # Define domain radii
    domain_radii = {
        'time': 0.004,
        'abs_pow': 0.006,
        'rel_pow': 0.008,
        'ai': 0.010
    }

    # Extract electrodes for each domain
    domain_electrodes = {}

    # Time domain electrodes
    time_electrodes = set()
    for feature in time_features:
        parts = feature.split('_')
        electrode = parts[-1]
        time_electrodes.add(electrode)
    domain_electrodes['time'] = time_electrodes

    # Absolute power domain electrodes
    abs_pow_electrodes = set()
    for feature in abs_pow_features:
        parts = feature.split('_')
        electrode = parts[-1]
        abs_pow_electrodes.add(electrode)
    domain_electrodes['abs_pow'] = abs_pow_electrodes

    # Relative power domain electrodes
    rel_pow_electrodes = set()
    for feature in rel_pow_features:
        parts = feature.split('_')
        electrode = parts[-1]
        rel_pow_electrodes.add(electrode)
    domain_electrodes['rel_pow'] = rel_pow_electrodes

    # Asymmetry index electrodes - these have a different structure (e.g., "ai_band_Ch1-Ch2")
    ai_electrodes = set()
    for feature in ai_features:
        parts = feature.split('_')
        electrode_pair = parts[-1]
        # Extract both electrodes from the pair
        ch_pair = electrode_pair.split('-')
        ai_electrodes.update(ch_pair)
    domain_electrodes['ai'] = ai_electrodes

    # Print electrode count for each domain
    for domain, electrodes in domain_electrodes.items():
        print(f"{domain} electrodes: {electrodes}")

    # Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the sensors (black dots) with their labels
    mne.viz.plot_sensors(info, show_names=True, axes=ax, show=False)

    # Update sensor label font to serif
    for text in ax.texts:
        text.set_fontfamily('serif')

    # Access the scatter plot of the sensors
    scatter = ax.collections[-1]

    # Get the 2D positions of the plotted sensors
    positions = scatter.get_offsets()

    # Add half-circles for all electrodes in each domain
    legend_elements = []
    for domain, electrodes in domain_electrodes.items():
        color = domain_colors[domain]
        radius = domain_radii.get(domain, 0.004)  # Use domain-specific radius

        # Draw half-circles for this domain
        for ch_name, pos in zip(channel_names, positions):
            if ch_name in electrodes:
                # Offset for multiple domains on same electrode
                offset = 0
                for d_index, (other_domain, other_electrodes) in enumerate(
                        domain_electrodes.items()):
                    if domain != other_domain and ch_name in other_electrodes:
                        if list(domain_electrodes.keys()).index(domain) > d_index:
                            offset += 0.002

                # Draw left half-circle (arc from 90 to 270 degrees)
                half_circle = Arc(pos,
                                  width=2*(radius + offset),  # Diameter
                                  height=2*(radius + offset),  # Diameter
                                  theta1=45, theta2=315,  # Left half
                                  color=color,
                                  linewidth=6,
                                  alpha=0.5)
                ax.add_patch(half_circle)

        # Add to legend with proper representation of half-circle and nicer label
        legend_elements.append(plt.Line2D(
            [0], [0], color=color, lw=2, label=domain_labels[domain]))

    # Add legend with improved appearance
    ax.legend(handles=legend_elements, loc='center', fontsize=12,
              framealpha=0.7, title="Feature Domains", title_fontsize=13,
              prop={'family': 'serif'})

    # Remove padding from figure
    fig.tight_layout(pad=0)

    # Print summary of feature distribution
    print(f"\nFeature distribution:")
    print(f"Time domain features: {len(time_features)}")
    print(f"Absolute power features: {len(abs_pow_features)}")
    print(f"Relative power features: {len(rel_pow_features)}")
    print(f"Asymmetry index features: {len(ai_features)}")

    return fig


def plot_features_by_frequency(selected_features, channel_names, info):
    """Plot selected features grouped by frequency bands"""
    # Define frequency band colors
    freq_colors = {
        # 'delta': 'purple',
        'theta': 'blue',
        'alpha-1': 'orange',
        'alpha-2': 'teal',
        'beta': 'green',
        'gamma': 'red',
    }

    # Define human-readable frequency band names for legend
    freq_labels = {
        # 'delta': 'Delta (0.5-4 Hz)',
        'theta': 'Theta (4-8 Hz)',
        'alpha-1': 'Alpha-1 (8-10 Hz)',
        'alpha-2': 'Alpha-2 (10-12 Hz)',
        'beta': 'Beta (12-30 Hz)',
        'gamma': 'Gamma (30-45 Hz)',
    }

    # Define frequency radii (to space out the arcs)
    freq_radii = {
        # 'delta': 0.004,
        'theta': 0.006,
        'alpha-1': 0.008,
        'alpha-2': 0.010,
        'beta': 0.012,
        'gamma': 0.014,
    }

    # Group features by frequency bands
    freq_features = {band: [] for band in freq_colors.keys()}

    for feature in selected_features:
        # Extract frequency band from feature name
        if feature.startswith(('abs_pow_', 'rel_pow_')):
            # Format: abs_pow_band_electrode or rel_pow_band_electrode
            parts = feature.split('_')
            band = parts[2]  # Extract the frequency band
            freq_features[band].append(feature)
        elif feature.startswith('ai_'):
            # Format: ai_band_electrodes
            parts = feature.split('_')
            band = parts[1]  # Extract the frequency band
            freq_features[band].append(feature)

    # Extract electrodes for each frequency band
    freq_electrodes = {}

    for band, features in freq_features.items():
        electrodes = set()
        for feature in features:
            if feature.startswith(('abs_pow_', 'rel_pow_')):
                parts = feature.split('_')
                electrode = parts[-1]
                electrodes.add(electrode)
            elif feature.startswith('ai_'):
                parts = feature.split('_')
                electrode_pair = parts[-1]
                # Extract both electrodes from the pair
                ch_pair = electrode_pair.split('-')
                electrodes.update(ch_pair)
        freq_electrodes[band] = electrodes

    # Print electrode count for each frequency band
    for band, electrodes in freq_electrodes.items():
        if electrodes:  # Only print non-empty sets
            print(f"{band} band electrodes: {electrodes}")

    # Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot the sensors (black dots) with their labels
    mne.viz.plot_sensors(info, show_names=True, axes=ax, show=False)

    # Update sensor label font to serif
    for text in ax.texts:
        text.set_fontfamily('serif')

    # Access the scatter plot of the sensors
    scatter = ax.collections[-1]

    # Get the 2D positions of the plotted sensors
    positions = scatter.get_offsets()

    # Add half-circles for all electrodes in each frequency band
    legend_elements = []
    for band, electrodes in freq_electrodes.items():
        if not electrodes:  # Skip empty electrode sets
            continue

        color = freq_colors[band]
        radius = freq_radii.get(band, 0.004)  # Use band-specific radius

        # Draw half-circles for this frequency band
        for ch_name, pos in zip(channel_names, positions):
            if ch_name in electrodes:
                # Offset for multiple bands on same electrode
                offset = 0
                for b_index, (other_band, other_electrodes) in enumerate(
                        freq_electrodes.items()):
                    if band != other_band and ch_name in other_electrodes:
                        if list(freq_electrodes.keys()).index(band) > b_index:
                            offset += 0.002

                # Draw left half-circle (arc from 90 to 270 degrees)
                half_circle = Arc(pos,
                                  width=2*(radius + offset),  # Diameter
                                  height=2*(radius + offset),  # Diameter
                                  theta1=45, theta2=315,  # Left half
                                  color=color,
                                  linewidth=6,
                                  alpha=0.5)
                ax.add_patch(half_circle)

        # Add to legend with proper representation of half-circle and nicer label
        legend_elements.append(plt.Line2D(
            [0], [0], color=color, lw=2, label=freq_labels[band]))

    # Add legend with improved appearance
    ax.legend(handles=legend_elements, loc='center', fontsize=12,
              framealpha=0.7, title="Frequency Bands", title_fontsize=13,
              prop={'family': 'serif'})

    # Remove padding from figure
    fig.tight_layout(pad=0)

    # Print summary of feature distribution by frequency band
    print(f"\nFeature distribution by frequency band:")
    for band, features in freq_features.items():
        if features:  # Only print non-empty lists
            print(f"{band} band features: {len(features)}")

    return fig


# Main code
if __name__ == "__main__":
    # Load the best classifier results
    best_results_path = '/home/jakub/projects/thesis/data/results/best_classif.csv'
    results_df = pd.read_csv(best_results_path)

    # Extract selected features
    selected_features_str = results_df['selected_features'].iloc[0]
    # Split the string by comma and strip whitespace and quotes
    selected_features = [f.strip(' "\'')
                         for f in selected_features_str.split(',')]

    # List of all electrode names
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    # Create an info object with the electrode names and set the standard 10-20 montage
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=256., ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Plot features by domain
    fig_domain = plot_features_by_domain(
        selected_features, channel_names, info)
    fig_domain.savefig(os.path.join(plots_dir, 'features_by_domain.pdf'))
    plt.show()

    # Plot features by frequency band
    fig_freq = plot_features_by_frequency(
        selected_features, channel_names, info)
    fig_freq.savefig(os.path.join(plots_dir, 'features_by_frequency.pdf'))
    plt.show()
