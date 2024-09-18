from common import Trial
import random
import numpy as np


def oversample(trials: list[Trial]):
    labels = map(lambda trial: trial.trial_label, trials)
    uniq_labels = list(set(labels))

    label_to_trial = {
        label: list(filter(lambda t: t.trial_label == label, trials)) for label in uniq_labels
    }

    max_len = max([len(i) for i in label_to_trial.values()])

    for label, trials_ in label_to_trial.items():
        random.shuffle(trials_)

        n_to_oversample = max_len - len(trials_)
        new_samples = trials_[:n_to_oversample]

        trials_.extend(new_samples)

    arr = np.array([i for i in label_to_trial.values()])
    arr = arr.flatten()

    return arr
