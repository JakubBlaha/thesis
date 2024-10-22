from common import Trial
import random
import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from src.models.cnn import FcCnn, FcCnnDataset
from torch import optim
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import tensor


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


def train_eval_pytorch_model(
        model, train_dataset, test_dataset, *, num_epochs=100, learning_rate=0.001, batch_size=32, last_epochs_avg=10):
    print("Train samples: ", len(train_dataset))
    print("Test samples: ", len(test_dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    train_acc = []
    test_acc = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(num_epochs)):
        # Train
        model.train()

        num_correct = 0
        num_samples = 0

        losses_ = []

        for data, targets in train_loader:
            t = tensor(data, dtype=torch.float32)

            scores = model.forward(t)
            loss = criterion(scores, targets)

            losses_.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

        train_losses.append(np.mean(losses_))
        train_acc.append(num_correct/num_samples)

        # Evaluate
        model.eval()

        with torch.no_grad():
            num_correct = 0
            num_samples = 0

            losses_ = []

            for data, targets in test_loader:
                scores = model.forward(tensor(data, dtype=torch.float32))
                loss = criterion(scores, targets)

                losses_.append(loss.item())

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            val_losses.append(np.mean(losses_))
            test_acc.append(num_correct/num_samples)

    plt.figure(figsize=(20, 5))
    plt.ylim(0, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.ylim(0, 1)
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.show()

    print(f"Avg test acc last {last_epochs_avg} epochs: ", np.mean(test_acc[-last_epochs_avg:]))
