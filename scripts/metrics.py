# Author: Jakub Bláha, xblaha36

"""
Machine Learning Metrics Calculator

This script calculates and outputs various machine learning metrics from a CSV file 
containing predicted and actual labels. It can also generate a confusion matrix 
visualization and save it as a PDF.

Usage:
    python metrics.py --file path/to/data.csv --title "Model Name"
    python metrics.py --latest

Input:
    - CSV file with 'predicted' and 'actual' columns
    - Optional title for the confusion matrix plot

Output:
    - Printed metrics including accuracy, balanced accuracy, Cohen's kappa,
      precision, recall, and F1 scores
    - Confusion matrix visualization saved as PDF (if title is provided)

Example:
    python metrics.py --file results.csv --title "Random Forest Model"
    python metrics.py --latest
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report,
                             cohen_kappa_score, precision_score,
                             recall_score, f1_score)

plots_dir = os.path.join(
    os.path.dirname(__file__),
    '..', 'data', 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Define the results directory path (same as in training.py)
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(data_dir, 'results')


def plot_confusion_matrix(conf_matrix, class_labels, title=None,
                          cmap='Blues', figsize=(5, 4),
                          annot=True, fmt='d'):
    """
    Plot a confusion matrix using heatmap.

    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        The confusion matrix data. Should be a square matrix where conf_matrix[i, j] 
        is the number of observations with actual class i predicted as class j.
    class_labels : list
        List of class labels.
    title : str, default='Confusion Matrix'
        Title for the plot.
    cmap : str, default='Blues'
        Colormap for the heatmap.
    figsize : tuple, default=(10, 8)
        Figure size.
    annot : bool, default=True
        If True, write the data value in each cell.
    fmt : str, default='d'
        String formatting code for the annotations.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis object.
    """
    # Set serif font for all text elements
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    # Convert to numpy array if not already
    conf_matrix = np.array(conf_matrix)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Draw the heatmap
    sns.heatmap(conf_matrix, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels,
                square=True, cbar=True, ax=ax)

    # Set labels and title with serif font
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)

    # Set tick labels to serif font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')

    # Tight layout to ensure everything fits
    plt.tight_layout()

    return fig, ax


def process_latest_result_file():
    """
    Process the most recent result file (alphabetically last) in the results directory.

    This function finds the latest CSV file in the results directory based on alphabetical 
    sorting, then processes it to calculate metrics and optionally generate a confusion matrix.

    Returns:
    --------
    str
        Path to the processed file, or None if no files were found
    """
    # List all CSV files in the results directory
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the results directory.")
        return None

    # Sort files alphabetically and get the last one
    csv_files.sort()
    latest_file = csv_files[-1]
    file_path = os.path.join(results_dir, latest_file)

    print(f"Processing the latest result file: {latest_file}")

    # Extract a title from the filename for the plot
    # Use the classifier name from the filename as the title
    title_parts = latest_file.split('_')
    if len(title_parts) > 1:
        # Use first part of filename as title
        title = title_parts[0].capitalize()
    else:
        title = "Latest Model"

    # Process the file using the main function
    main(file_path, title)

    return file_path


def main(file_path, title=None):
    # Read CSV file with columns "predicted" and "actual"
    df = pd.read_csv(file_path, comment='#')
    y_pred = df['predicted']
    y_true = df['actual']

    # Calculate basic accuracy metrics
    accuracy = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Calculate confusion matrix and detailed classification report
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, digits=3)

    # Calculate macro and weighted averaged precision, recall and F1 scores
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Output results
    print("=== Machine Learning Metrics ===\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {bal_accuracy:.3f}")
    print(f"Cohen's Kappa: {kappa:.3f}\n")

    print("Confusion Matrix:")
    print(conf_matrix, "\n")

    print("Classification Report:")
    print(class_report)

    print("Macro Averages:")
    print(f"  Precision: {macro_precision:.3f}")
    print(f"  Recall:    {macro_recall:.3f}")
    print(f"  F1 Score:  {macro_f1:.3f}\n")

    print("Weighted Averages:")
    print(f"  Precision: {weighted_precision:.3f}")
    print(f"  Recall:    {weighted_recall:.3f}")
    print(f"  F1 Score:  {weighted_f1:.3f}")

    # Only generate confusion matrix if title is provided
    if title is None:
        print("\nNo title provided. Skipping confusion matrix generation.")
        return

    # Generate confusion matrix pdf
    nice_labels = {
        'HI_GAD': 'GAD',
        'HI_SAD': 'SAD',
        'CONTROL': 'Control'
    }

    # Define the full label order
    label_order = ['HI_GAD', 'HI_SAD', 'CONTROL']

    # Only use labels that are present in the data
    present_labels = [label for label in label_order
                      if label in df['actual'].unique() or label in df
                      ['predicted'].unique()]
    if not present_labels:
        print("No known labels found in the data. Skipping confusion matrix generation.")
        return

    labels = [nice_labels[label] for label in present_labels]

    # Compute confusion matrix with explicit label order for sklearn
    conf_matrix = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plot_confusion_matrix(
        conf_matrix, labels, title=title)
    fname = f'conf_matrix__{title.lower().replace(" ", "_")}.pdf'
    path = os.path.abspath(os.path.join(plots_dir, fname))

    print()
    print(f"Saving confusion matrix plot to: {path}")
    print("  " + path)

    plt.savefig(os.path.join(plots_dir, fname))
