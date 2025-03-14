# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

plot_dir = os.path.join(
    os.path.dirname(__file__),
    '..', '..', 'data', 'plots')
os.makedirs(plot_dir, exist_ok=True)


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


# %%
# LSTM final enhanced
labels = ['GAD', 'SAD', 'Control']
conf_matrix = [
    [1491,   83,  744],
    [45, 1964,  391],
    [876, 1150, 1335]
]

fig, ax = plot_confusion_matrix(conf_matrix, labels)
plt.savefig(os.path.join(plot_dir, 'conf_matrix_lstm_final_enhanced.pdf'))

# %%
# LSTM final
conf_matrix = [
    [1411,  429,  478],
    [78, 2056,  266],
    [1016, 2063,  282]
]

fig, ax = plot_confusion_matrix(conf_matrix, labels)
plt.savefig(os.path.join(plot_dir, 'conf_matrix_lstm_final.pdf'))

# %%
# CNN final
conf_matrix = [
    [41,  3,  40],
    [0, 59,  23],
    [14, 20,  88]
]

fig, ax = plot_confusion_matrix(conf_matrix, labels)
plt.savefig(os.path.join(plot_dir, 'conf_matrix_cnn_final.pdf'))
