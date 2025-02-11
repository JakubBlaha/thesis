# %%
from utils import DatasetBuilder, LabelingScheme, DaspsLabeling
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib

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
            nn.BatchNorm2d(40, affine=True),
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
            nn.BatchNorm2d(160, affine=True),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Pool 3
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Layer 5
            nn.Conv2d(in_channels=160, out_channels=160,
                      kernel_size=(7, 1), stride=(7, 1)),
            nn.BatchNorm2d(160, affine=True),
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
    

import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_classes, seq_len=None, dropout=0.5, num_channels=14):  # num_channels added
        super(EEGNet, self).__init__()

        self.model = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=num_channels, out_channels=20, kernel_size=5, padding=2),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, padding=2),
            nn.BatchNorm1d(40),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2),

            # Layer 3
            nn.Conv1d(in_channels=40, out_channels=80, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 4
            nn.Conv1d(in_channels=80, out_channels=160, kernel_size=3, padding=1),
            nn.BatchNorm1d(160),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2),

            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),

            # Flatten
            nn.Flatten(),

            # Linear Layer
            nn.Linear(160, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


# class ConvNetCustom(nn.Module):
#     def __init__(self, seq_len, num_classes=0, dropout=0.0):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.ZeroPad2d((15, 15, 0, 0)),
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 31), stride=(1, 1), padding=0),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),

#             nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(2, 1), stride=(2, 1), padding=0),
#             nn.BatchNorm2d(40, affine=False),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

#             nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(1, 21), stride=(1, 1)),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

#             nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(1, 11), stride=(1, 1)),
#             nn.BatchNorm2d(160, affine=False),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

#             nn.Conv2d(in_channels=160, out_channels=160, kernel_size=(7, 1), stride=(7, 1)),
#             nn.BatchNorm2d(160, affine=False),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

#             nn.Flatten(start_dim=1),

#             nn.Linear(4 * 160, num_classes),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.model):
#             x = layer(x)
#             print(f"Layer {i}: {
#                   layer.__class__.__name__}, Output Shape: {x.shape}")
#         return x


def compile_model(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def evaluate_model(model, test_loader, criterion):
    model.eval()
    num_correct = 0
    num_samples = 0
    losses_ = []

    with torch.no_grad():
        for data, targets in test_loader:
            scores = model.forward(data)
            loss = criterion(scores, targets)
            losses_.append(loss.detach())
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    avg_loss = torch.stack(losses_).mean().item()
    accuracy = num_correct / num_samples
    return avg_loss, accuracy

def train_eval_pytorch_model(
        model, train_dataset, test_dataset, *, num_epochs=100,
        learning_rate=0.001, batch_size=32, last_epochs_avg=10,
        enable_profiling=False):
    print("Train samples: ", len(train_dataset))
    print("Test samples: ", len(test_dataset))

    criterion, optimizer = compile_model(model, learning_rate)

    train_losses = []
    val_losses = []

    train_acc = []
    test_acc = []

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    if enable_profiling:
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True)
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context as prof:
        for epoch in tqdm(range(num_epochs)):
            # Train
            model.train()

            num_correct = 0
            num_samples = 0

            losses_ = []

            for data, targets in train_loader:
                with record_function("model_inference"):
                    scores = model.forward(data)
                loss = criterion(scores, targets)

                losses_.append(loss.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

            train_losses.append(torch.stack(losses_).mean().item())
            train_acc.append(num_correct/num_samples)

            # Evaluate
            val_loss, val_accuracy = evaluate_model(model, test_loader, criterion)
            val_losses.append(val_loss)
            test_acc.append(val_accuracy)

    if enable_profiling:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

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

    print(f"Max test acc: ", max(test_acc))
    print(f"Max accuracy epoch: ", np.argmax(test_acc))


seglen_to_params = {
    1: { # 0.61
        "num_epochs": 40,
        "learning_rate": 0.00001,
        "batch_size": 16,
        "dropout": 0.35,
    },
    2: { # 0.56
        "num_epochs": 60,
        "learning_rate": 0.00001,
        "batch_size": 32,
        "dropout": 0.35,
    },
    3: { # 0.633
        "num_epochs": 50,
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
    5: { # 0.553
        "num_epochs": 50,
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
    10: { # 0.59
        "num_epochs": 100,
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
    15: { # 0.597
        "num_epochs": 100,
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
    30: { # 0.589
        "num_epochs": 100,
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
}

seglen = 3
use_gpu = True


if __name__ == "__main__":
    # Seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Setup HW acceleration
    if torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Build dataset
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    test_subj_ids = [
        8, 9, 10,  # Low DASPS
        1, 2, 3, 4, 5, 6, 7, # High DASPS
        *range(101, 106), # Low SAD
        *range(401, 408)  # High SAD
    ]

    # Remove half of test subjects randomly
    # test_subj_ids = np.random.choice(test_subj_ids, len(test_subj_ids) // 2, replace=False)

    train, test = builder.build_deep_datasets_train_test(
        seglen=seglen, insert_ch_dim=False, test_subj_ids=test_subj_ids, device=device)

    data, _ = train[0]
    seq_len = data.shape[-1]
    print("Seq len: ", train[0][0].shape[-1])

    params = seglen_to_params.get(seglen)

    if params is None:
        raise ValueError(f"No parameters defined for seglen: {seglen}")

    # Print labels
    # print(list(train.labels.cpu().numpy()))


    model = EEGNet(seq_len=seq_len, num_classes=3, dropout=params["dropout"])
    model.to(device)

    train_eval_pytorch_model(
        model, train, test, num_epochs=params["num_epochs"], learning_rate=params["learning_rate"],
        enable_profiling=False, batch_size=params["batch_size"])

    # Evaluate the model on the test set
    test_loader = DataLoader(test, batch_size=params["batch_size"], shuffle=False)
    test_loss, test_accuracy = evaluate_model(model, test_loader, nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

