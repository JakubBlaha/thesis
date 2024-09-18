import torch
import torch.nn as nn
from torch.utils.data import Dataset
from common import Trial, TrialLabel


class CNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(16 * 8 * 8, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class CnnDataset(Dataset):
    def __init__(self, trials: list[Trial]) -> None:
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]

        return trial.power_matrices[:, :, :], trial.trial_label.value
