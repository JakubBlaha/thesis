# %%
from utils import DatasetBuilder, LabelingScheme, DaspsLabeling
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import tensor
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Inspired by the following
# https://github.com/CNN-for-EEG-classification/CNN-EEG/blob/main/convNet.py


class ConvNet(nn.Module):
    def __init__(self, seq_len, num_classes, dropout=0.0):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.ZeroPad2d((15, 15, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=20,
                      kernel_size=(1, 31), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Conv2d(in_channels=20, out_channels=40,
                      kernel_size=(2, 1), stride=(2, 1), padding=0),
            nn.BatchNorm2d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            # Layer 3
            nn.Conv2d(in_channels=40, out_channels=80,
                      kernel_size=(1, 21), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Pool 2
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(in_channels=80, out_channels=160,
                      kernel_size=(1, 11), stride=(1, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Pool 3
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Layer 5
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=(7, 1), stride=(7, 1)),
            nn.BatchNorm2d(160, affine=False),
            nn.LeakyReLU(),

            # Pool 4
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Flatten layer
            nn.Flatten(start_dim=1),

            # Linear Layer
            nn.Linear(5280, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # print(x.shape)

        for i, layer in enumerate(self.model):
            x = layer(x)
            # print(
            #     f"Layer {i}: {layer.__class__.__name__}, Output Shape: {x.shape}")
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
            print(f"Layer {i}: {
                  layer.__class__.__name__}, Output Shape: {x.shape}")
        return x


def compile_model(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_eval_pytorch_model(
        model, train_dataset, test_dataset, *, num_epochs=100,
        learning_rate=0.001, batch_size=32, last_epochs_avg=10):
    print("Train samples: ", len(train_dataset))
    print("Test samples: ", len(test_dataset))

    criterion, optimizer = compile_model(model, learning_rate)

    train_losses = []
    val_losses = []

    train_acc = []
    test_acc = []

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in tqdm(range(num_epochs)):
        # Train
        model.train()

        num_correct = 0
        num_samples = 0

        losses_ = []

        for data, targets in train_loader:
            t = data.to(device)
            targets = targets.to(device)

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
                t = data.to(device)
                targets = targets.to(device)

                scores = model.forward(t)
                loss = criterion(scores, targets)

                losses_.append(loss.item())

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            val_losses.append(np.mean(losses_))
            test_acc.append(num_correct/num_samples)

    train_acc = [float(i.cpu().detach().numpy()) for i in train_acc]
    test_acc = [float(i.cpu().detach().numpy()) for i in test_acc]

    # Plotting with seaborn
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    configs = [
        (axs[0], {"Train Loss": train_losses, "Validation Loss": val_losses}, "Loss"),
        (axs[1], {"Train Accuracy": train_acc, "Test Accuracy": test_acc}, "Accuracy")
    ]
    for ax, data_dict, title in configs:
        for label, values in data_dict.items():
            sns.lineplot(x=range(len(values)), y=values, ax=ax, label=label)
        ax.set(ylim=(0, 1))
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("Epochs")

    plt.show()

    # Max accuracy
    if len(test_acc) > 5:
        max_test_acc = max(test_acc[5:])
        max_index = 5 + np.argmax(test_acc[5:])
        print(f"Max test acc after 5 epochs: ", max_test_acc)
        print(f"Max accuracy epoch: ", max_index)
    else:
        print("Not enough epochs to compute max test accuracy after 5 epochs.")


if __name__ == "__main__":
    # Setup HW acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Build dataset
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    train, test = builder.build_deep_datasets_train_test(10)

    data, label = train[0]

    channels, n_electrodes, seq_len = data.shape

    print("Seq len: ", seq_len)

    model = ConvNet(seq_len=seq_len, num_classes=3, dropout=0)
    model.to(device)

    train_eval_pytorch_model(
        model, train, test, num_epochs=10, learning_rate=0.00001)

    # torch.save(model.state_dict(), 'trained_model.pth')
