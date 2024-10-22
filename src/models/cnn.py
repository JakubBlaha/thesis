import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from common import CHANNEL_NAMES, Trial, TrialLabel
from torch import tensor
import torch


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


class RatiosCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(8 * 4 * 10 // 4 // 4, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class RatiosCnnDataset(Dataset):
    def __init__(self, trials: list[Trial]) -> None:
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]

        extended_pow_ratios = trial.pow_ratios.copy()

        extended_pow_ratios = np.insert(extended_pow_ratios, 5, extended_pow_ratios[:, 4, :], axis=1)
        extended_pow_ratios = np.insert(extended_pow_ratios, 6, extended_pow_ratios[:, 4, :], axis=1)
        extended_pow_ratios = np.insert(extended_pow_ratios, 7, extended_pow_ratios[:, 4, :], axis=1)

        return extended_pow_ratios, trial.trial_label.value


class FcCnn(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(16 * 16 * 10 // 4 // 4, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class FcCnnDataset(Dataset):
    def __init__(self, trials: list[Trial]) -> None:
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]

        fc_matrix = np.pad(trial.fc_matrix, ((0, 0), (1, 1), (1, 1)), 'constant')

        return fc_matrix, trial.trial_label.value


class AllFeatsCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),  # Try leaky relu
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),  # Try leaky relu
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
        )

        self.fc = nn.Linear(2 * 3 * 8, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class AllFeatsDataset(Dataset):
    def __init__(self, trials: list[Trial]) -> None:
        self.trials = trials

        feats = self.trials[0].features
        time_feats = [i for i in feats if i.startswith('time_')]
        time_feats = set(['_'.join(feat_name.split('_')[:-1]) for feat_name in time_feats])

        self._time_feat_names = time_feats

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]

        # Build feature matrix
        m = []

        for chname in CHANNEL_NAMES:
            ch_feats = []

            for time_feat_name in self._time_feat_names:
                ch_feats.append(trial.features[time_feat_name + "_" + chname])

            m.append(ch_feats)

        return tensor([m]), trial.trial_label.value


# class DeepCnn(nn.Module)
#     def __init__(self, n_classes: int, seq_len: int) -> None:
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(8),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         self.fc = nn.Linear((seq_len // 2 // 2) * (14 // 2 // 2) * 8, n_classes)

#     def forward(self, x):
#         x = self.model(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc(x)

#         return x


class DeepCnn(nn.Module):
    def __init__(self, n_classes: int, seq_len: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear((seq_len // 2 // 2) * (14 // 2 // 2) * 8, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout2d(x, 0.5)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.dropout2d(x, 0.5)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # x = self.conv3(x)
        # x = nn.functional.relu(x)
        # x = nn.functional.dropout2d(x, 0.5)
        # x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # print(x.shape)

        # x = x.reshape(x.shape[0], -1)
        # x = x.flatten()
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)

        return x


class DeepCnnDataset(Dataset):
    def __init__(self, trials: list[Trial], max_len: int, channels: list[int] = list(range(0, 14))) -> None:
        self.trials = trials
        self.max_len = max_len
        self.channels = channels

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, index):
        trial = self.trials[index]

        return tensor(trial.epoch.get_data()[:, self.channels, :self.max_len]), trial.trial_label.value


class ConvNet(nn.Module):
    def __init__(self, seq_len, num_classes=0, dropout=0.0):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.ZeroPad2d((15, 15, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 31), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            # Layer 3
            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 21), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Pool 2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # Layer 4
            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(1, 11), stride=(1, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Pool 3
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Layer 5
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(7, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),

            # Pool 4
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Flatten layer
            nn.Flatten(start_dim=1),

            # Linear Layer
            nn.Linear(4 * 160, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            # print(f"Layer {i}: {layer.__class__.__name__}, Output Shape: {x.shape}")
        return x


class CnnNet2(nn.Module):
    def __init__(self, seq_len, n_cls, dropout=0.0):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 11), stride=1, padding=0),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 11), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 11), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 11), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Flatten(),
            nn.Linear(200 * 4, n_cls)
        )

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            # print(f"Layer {i}: {layer.__class__.__name__}, Output Shape: {x.shape}")
        return x


class ConvNetCustom(nn.Module):
    def __init__(self, seq_len, num_classes=0, dropout=0.0):
        super().__init__()

        self.model = nn.Sequential(
            nn.ZeroPad2d((15, 15, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 31), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 21), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(1, 11), stride=(1, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(7, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            nn.Flatten(start_dim=1),

            nn.Linear(4 * 160, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"Layer {i}: {layer.__class__.__name__}, Output Shape: {x.shape}")
        return x
