from dasps_preprocess import DaspsPreprocessor, Severity

import matplotlib.pyplot as plt
import numpy as np


def plot_severities(situations):
    severities = {
        Severity.NORMAL: 0,
        Severity.LIGHT: 0,
        Severity.MODERATE: 0,
        Severity.SEVERE: 0,
    }

    for participant in situations:
        print(participant)
        # print(participant, participant.get_severity())

        severity = participant.get_sam_label()
        severities[severity] += 2

    print(severities)

    points = [[sit.valence, sit.arousal] for sit in situations]
    points = np.array(points)
    x, y = zip(*points)

    n_points = {}

    for point in points:
        key = f"{point[0]}, {point[1]}"

        n_points[key] = n_points.get(key, 0) + 2

    labels = [f"{point[0]}, {point[1]}" for point in points]

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=0)

    for i, txt in enumerate(labels):
        key = f"{x[i]}, {y[i]}"
        ax.annotate(n_points[key], (x[i], y[i]), ha='center', va='center')

    ax.axhline(y=4.5, color='red')
    ax.axvline(x=5.5, color='red')

    # Highlight LVHA
    rect = plt.Rectangle((0, 4.5), 5.5, 5.5, color='gray', alpha=0.2)
    ax.add_patch(rect)

    # Highlight severe
    rect = plt.Rectangle((0, 6.5), 2.5, 3, color='red', alpha=0.5)
    ax.add_patch(rect)

    # Highlight moderate
    rect = plt.Rectangle((2.5, 5.5), 2, 1, color='orange', alpha=0.5)
    ax.add_patch(rect)

    # Highlight light
    rect = plt.Rectangle((4.5, 4.5), 1, 1, color='yellow', alpha=0.5)
    ax.add_patch(rect)

    # Axis labels
    plt.xlabel('Valence')
    plt.ylabel('Arousal')

    plt.show()


if __name__ == '__main__':
    situations = DaspsPreprocessor.get_situations()
    plot_severities(situations)
