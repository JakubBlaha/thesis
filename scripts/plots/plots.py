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
results_path = os.path.join(script_path, "..", "data", "results")

plots_path = os.path.join(script_path, "../data/plots")

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


def gen_seglens_plot():
    # These need to be replaced in case the chart needs to be generated
    fnames = [
        'classif_20250301_131128_mode-both_seglens-30_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250301_133310_mode-both_seglens-15_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250301_134642_mode-both_seglens-10_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250301_142502_mode-both_seglens-5_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250301_154354_mode-both_seglens-3_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250301_182308_mode-both_seglens-2_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv',
        'classif_20250302_120211_mode-both_seglens-1_domains-rel_pow-conn-ai-time-abs_pow_label-ham_cv-logo_os.csv']

    paths = [os.path.join(results_path, fname) for fname in fnames]

    df = pd.concat([pd.read_csv(path) for path in paths])
    df.drop(
        columns=["macro_f1", "macro_precision", "macro_recall",
                 "best_params", "num_selected_features",
                 "selected_features"],
        inplace=True)
    df["seglen"] = df["seglen"].astype(str)

    # Replace classifier names with user-friendly names
    df["classifier"] = df["classifier"].apply(get_display_name)

    # Sort by seglen
    df["seglen"] = pd.Categorical(
        df["seglen"],
        categories=["1", "2", "3", "5", "10", "15", "30"],
        ordered=True)

    print(df.columns)

    # Calculate average accuracy for each segment length
    avg_by_seglen = df.groupby('seglen')['mean_accuracy'].mean().reset_index()
    # Add classifier column for the legend
    avg_by_seglen['classifier'] = 'Average'

    # Create the main plot with the classifiers
    g = sns.relplot(
        data=df,
        x="seglen",
        y="mean_accuracy",
        hue="classifier",
        kind="line",
        aspect=1.5,
        height=5,
        markers=True,
        style="classifier",
        dashes=False,
        legend="brief",
    )

    # Add the average line
    ax = g.axes[0, 0]
    ax.plot(avg_by_seglen['seglen'].cat.codes, avg_by_seglen['mean_accuracy'],
            color='black', linestyle='--', linewidth=2, marker='o',
            label='Average')

    # Set xticks
    g.set(xticks=df["seglen"].unique())
    g.set_axis_labels("Segment length [s]", "Accuracy")
    # g.fig.suptitle("Accuracy to segment length by classifier", y=1.02)

    # Move legend to bottom left
    g._legend.remove()
    g.figure.legend(
        loc='lower right', bbox_to_anchor=(0.925, 0.12),
        frameon=True)

    plt.tight_layout()

    # Print seglens to average accuracy
    print(avg_by_seglen)

    # Save plot
    plot_path = os.path.join(plots_path, "seglens_plot.pdf")
    plt.savefig(plot_path)


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

    # Save plot
    plot_path = os.path.join(plots_path, "feature_significance.pdf")
    plt.savefig(plot_path)

    # Print top 10 most frequent features with classifier breakdown
    print("\nTop 10 Most Significant Features:")
    for i, (feature, counts) in enumerate(pretty_features[:10]):
        total = sum(counts.values())
        classifier_breakdown = ", ".join(
            [f"{c}: {counts[c]}" for c in counts if counts[c] > 0])
        print(f"{feature}: {total} times ({classifier_breakdown})")


def generate_feature_table_latex(csv_data):
    """Generate a LaTeX table of features ranked by inclusion frequency with two columns."""
    # Read CSV data into a DataFrame
    df = pd.read_csv(io.StringIO(csv_data))

    # Extract all features from the selected_features column
    all_features = []
    for features in df['selected_features']:
        # Split the feature string into a list and clean it
        feature_list = [f.strip() for f in features.split(',')]
        all_features.extend(feature_list)

    # Count frequency of each feature
    feature_counts = Counter(all_features)

    # # Sort features by frequency
    # sorted_features = sorted(
    #     feature_counts.items(),
    #     key=lambda x: x[1],
    #     reverse=True)

    # Process feature names to make them prettier
    pretty_features = []
    for feature, count in feature_counts.items():
        pretty_name = prettify_feature_name(feature)
        pretty_features.append((pretty_name, count))

    # Calculate midpoint to split into two columns
    midpoint = len(pretty_features) // 2
    if len(pretty_features) % 2 != 0:
        midpoint += 1  # Ensure first column gets the extra item if odd number

    first_column = pretty_features[:midpoint]
    second_column = pretty_features[midpoint:]

    # Pad second column with empty rows if needed
    while len(second_column) < len(first_column):
        second_column.append(("", ""))

    # Create LaTeX table with two columns of features
    latex_table = "\\textbf{Feature} & \\textbf{Freq.} & \\textbf{Feature} & \\textbf{Freq.} \\\\\n"
    latex_table += "\\hline\n"

    for i in range(len(first_column)):
        feature1, count1 = first_column[i]
        feature1_escaped = feature1.replace('_', '\\_')

        # Check if we have a valid second feature
        if i < len(second_column) and second_column[i][0]:
            feature2, count2 = second_column[i]
            feature2_escaped = feature2.replace('_', '\\_')
            latex_table += f"{feature1_escaped}       & {count1}       & {
                feature2_escaped}       & {count2}       \\\\\n"
        else:
            latex_table += f"{feature1_escaped} & {count1} & & \\\\\n"

    latex_table += "\\hline\n"

    # Save LaTeX table to CSV
    table_path = os.path.join(plots_path, "feature_table.tex")
    with open(table_path, 'w') as f:
        f.write(latex_table)


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
                    band.capitalize()}   ({electrode1} -{electrode2}) "

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
        generate_feature_table_latex(csv_data)
    else:
        print(
            f"Warning: Feature significance data not found at {feature_csv_path}")


def gen_plots():
    os.makedirs(plots_path, exist_ok=True)

    gen_seglens_plot()
    gen_feature_significance_plot()


gen_plots()
