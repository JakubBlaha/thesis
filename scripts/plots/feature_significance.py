# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import io
import numpy as np
from collections import Counter

plt.rcParams['font.family'] = 'serif'

script_path = os.path.dirname(os.path.realpath(__file__))
results_path = os.path.join(script_path, "../..", "data", "results")

CLASSIFIER_NAMES = {
    'svm-rbf': 'SVM (RBF kernel)',
    'svm-lin': 'SVM (Linear kernel)',
    'svm-poly': 'SVM (Polynomial kernel)',
    'rf': 'Random Forest',
    'knn': 'K-Nearest Neighbors',
    'mlp': 'Multi-Layer Perceptron',
}


def get_display_name(classifier_name):
    """Return a friendly display name for a classifier if available, otherwise return the original name."""
    return CLASSIFIER_NAMES.get(classifier_name, classifier_name)


def plot_feature_significance(csv_data):
    # Read CSV data into a DataFrame
    df = pd.read_csv(io.StringIO(csv_data))

    # Create a dictionary to track features selected by each classifier
    feature_classifier_counts = {}

    # Process each row in the dataframe
    for _, row in df.iterrows():
        classifier = row['classifier']
        # Get display name for the classifier
        display_classifier = get_display_name(classifier)

        # Split the feature string into a list and clean it
        features = [f.strip() for f in row['selected_features'].split(',')]

        # Count each feature for this classifier
        for feature in features:
            if feature not in feature_classifier_counts:
                feature_classifier_counts[feature] = Counter()
            feature_classifier_counts[feature][display_classifier] += 1

    # Sort features by total count (descending) - using frequency as proxy for significance
    # More frequently selected features are assumed to have lower p-values (higher significance)
    sorted_features = sorted(
        feature_classifier_counts.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )

    # Prettify feature names and keep counts
    pretty_features = []
    for feature, counts in sorted_features:
        pretty_name = prettify_feature_name(feature)
        pretty_features.append((pretty_name, counts))

    # Extract feature names and counts for plotting
    features, classifier_counts = zip(*pretty_features)

    # Get all unique classifiers
    all_classifiers = set()
    for counts in classifier_counts:
        all_classifiers.update(counts.keys())
    all_classifiers = sorted(all_classifiers)

    # Create figure
    plt.figure(figsize=(7, 10))

    # Features are displayed from bottom to top in order of increasing significance
    # (most significant/lowest p-value at the top)
    reversed_features = list(reversed(features))

    # Create a data array for stacked bars
    data = np.zeros((len(features), len(all_classifiers)))
    for i, counts in enumerate(classifier_counts):
        for j, classifier in enumerate(all_classifiers):
            data[i, j] = counts.get(classifier, 0)

    # Reverse data for plotting (so highest significance is at the top)
    reversed_data = np.flip(data, axis=0)

    # Plot stacked bars
    bottom = np.zeros(len(features))
    for i, classifier in enumerate(all_classifiers):
        plt.barh(
            reversed_features,
            reversed_data[:, i],
            left=bottom,
            height=0.5,
            label=classifier
        )
        bottom += reversed_data[:, i]

    # Customize plot with bigger text
    plt.xlabel('Frequency of Selection', fontsize=14, loc='left')
    plt.yticks(fontsize=10)  # Increase y-axis label size
    plt.xticks(fontsize=10)  # Increase x-axis label size
    plt.title('Feature Significance (ordered by selection frequency)', fontsize=14)

    # Add grid lines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Ensure x-axis has appropriate tick marks
    max_count = max([sum(counts.values()) for counts in classifier_counts])
    plt.xlim(0, max_count * 1.05)
    plt.ylim(-0.5, len(features) - 0.5)

    tick_interval = 1
    plt.xticks(range(0, int(max_count) + tick_interval, tick_interval))

    # Add legend
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05))

    plt.tight_layout()

    # Show plot instead of saving
    plt.show()


def prettify_feature_name(feature):
    """Convert feature names to a more readable format."""
    # Handle connectivity features (ai_band_electrode1-electrode2)
    if feature.startswith('ai_'):
        parts = feature.split('_', 2)  # Split into at most 3 parts
        if len(parts) >= 2:
            band = parts[1]
            electrodes = parts[2] if len(parts) > 2 else ""

            electrode_parts = electrodes.split('-')
            if len(electrode_parts) == 2:
                electrode1, electrode2 = electrode_parts
                return f"Asym. Index - {
                    band.capitalize()}    ({electrode1}  -{electrode2}) "

    # Handle relative and absolute power features
    if feature.startswith('rel_pow_') or feature.startswith('abs_pow_'):
        parts = feature.split('_', 2)  # Split into at most 3 parts
        if len(parts) == 3:
            # Reconstruct the domain (rel_pow or abs_pow)
            domain = parts[0] + '_' + parts[1]
            rest = parts[2]

            # Further split the rest part in case it's 'band_electrode'
            rest_parts = rest.split('_', 1)
            band = rest_parts[0]
            electrode = rest_parts[1] if len(rest_parts) > 1 else ""

            domain_name = "Rel. power" if domain == 'rel_pow' else "Abs. power"
            return f"{domain_name} - {band.capitalize()} ({electrode})"

    # Handle time domain features
    if feature.startswith('time_'):
        # Split by last underscore
        parts = feature.rsplit('_', 1)
        if len(parts) == 2:
            measure = parts[0]
            electrode = parts[1]
            measure = measure.replace('time_', '', 1)
            measure_name = expand_time_measure(measure)
            return f"{measure_name} ({electrode})"

    return feature


def expand_time_measure(measure):
    """Expand time domain measure abbreviations."""
    measure_map = {
        'hjorth_mobility': 'Hjorth mobility',
        'hjorth_complexity': 'Hjorth complexity',
        'app_entropy': 'Approximate entropy',
        'line_length': 'Line length',
        'higuchi_fd': 'Higuchi fractal dimension',
        'hurst_exp': 'Hurst exponent',
    }
    return measure_map.get(measure, measure.replace('_', ' '))


def gen_feature_significance_plot():
    feature_csv_path = os.path.join(
        results_path,
        "../../data/results/classif_20250303_172740_mode-both_seglens-15_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv")
    if os.path.exists(feature_csv_path):
        with open(feature_csv_path, 'r') as f:
            csv_data = f.read()
        plot_feature_significance(csv_data)
    else:
        print(
            f"Warning: Feature significance data not found at {feature_csv_path}")


gen_feature_significance_plot()
