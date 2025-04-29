# Author: Jakub Bláha, xblaha36

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(
        y_test_all, y_prob_all, unique_labels, int_to_label, classif: str,
        seglen, num_selected_features, results_dir):
    """
    Plot ROC curves for a multi-class classification problem.

    Parameters
    ----------
    y_test_all : list
        List of test label arrays from cross-validation folds.
    y_prob_all : list
        List of probability prediction arrays from cross-validation folds.
    unique_labels : array
        Unique class labels.
    int_to_label : dict
        Mapping from integer labels to string labels.
    classif : str
        Classifier name (for title and filename).
    seglen : int
        Segment length (for title and filename).
    num_selected_features : int
        Number of selected features (for filename).
    results_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved ROC curve plot.
    """
    # Set serif font for all text elements
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']

    plt.figure(figsize=(6, 5.5))

    # Concatenate all predictions from cross-validation folds
    y_test_concat = np.concatenate([y for y in y_test_all])
    y_prob_concat = np.concatenate([p for p in y_prob_all])

    # For each class, calculate ROC
    for i, class_idx in enumerate(unique_labels):
        # One-vs-rest approach
        y_true_binary = (y_test_concat == class_idx).astype(int)
        y_score = y_prob_concat[:, i]

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)

        # Map label names for display
        label_name = int_to_label[class_idx]
        if label_name == 'HI_SAD':
            label_name = 'SAD'
        elif label_name == 'HI_GAD':
            label_name = 'GAD'
        elif label_name == 'CONTROL':
            label_name = 'Control'

        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2,
                 label=f'ROC curve (class {label_name}, AUC = {roc_auc:.2f})')

    # Add a random classifier line for reference
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Format plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {classif.upper()} (seglen={seglen})')
    plt.legend(loc="lower right")

    # Save the plot
    timestamp = str(int(datetime.datetime.now().timestamp()))
    roc_filename = os.path.join(
        results_dir,
        f"roc_{timestamp}_{classif}_seglen_{seglen}_k_{num_selected_features}.pdf"
    )
    plt.savefig(roc_filename, dpi=300)
    plt.show()

    return roc_filename
