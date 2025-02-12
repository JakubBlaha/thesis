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
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn

# Inspired by the following
# https://github.com/CNN-for-EEG-classification/CNN-EEG/blob/main/convNet.py


class EEGNet(nn.Module):
    def __init__(self, num_classes, dropout=0.5, num_channels=14):  # num_channels added
        super(EEGNet, self).__init__()

        # 0.617

        n_feats_a = 20
        n_feats_b = 40
        n_feats_c = 80
        n_feats_d = 160

        self.model = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=num_channels, out_channels=n_feats_a, kernel_size=7, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_a),
            nn.Dropout(p=dropout),

            # Layer 2
            nn.Conv1d(in_channels=n_feats_a, out_channels=n_feats_b, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_b),
            nn.AvgPool1d(kernel_size=2),

            # Layer 3
            nn.Conv1d(in_channels=n_feats_b, out_channels=n_feats_c, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),

            # Layer 4
            nn.Conv1d(in_channels=n_feats_c, out_channels=n_feats_d, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_feats_d),
            nn.AvgPool1d(kernel_size=2),

            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1),

            # Flatten
            nn.Flatten(),

            # Linear Layer
            nn.Linear(n_feats_d, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


def compile_model(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def evaluate_model(model, test_loader, criterion):
    model.eval()

    all_predictions = []
    all_targets = []
    losses_ = []

    with torch.no_grad():
        for data, targets in test_loader:
            scores = model.forward(data)
            loss = criterion(scores, targets)
            losses_.append(loss.detach())
            _, predictions = scores.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = torch.stack(losses_).mean().item()
    
    # Calculate accuracy per label
    label_accuracies = {}
    for label in set(all_targets):
        label_indices = [i for i, target in enumerate(all_targets) if target == label]
        label_predictions = [all_predictions[i] for i in label_indices]
        correct_predictions = sum(1 for pred in label_predictions if pred == label)
        accuracy = correct_predictions / len(label_predictions) if len(label_predictions) > 0 else 0
        label_accuracies[label] = accuracy
    
    return avg_loss, all_predictions, all_targets, label_accuracies


def train_model(
        model, train_dataset, test_dataset, *, max_epochs=100,
        learning_rate=0.001, batch_size=32,
        enable_profiling=False, patience=10, min_epochs=30):
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

    best_test_acc = 0
    epochs_no_improve = 0
    state_dict = None

    with profiler_context as prof:
        for epoch in tqdm(range(max_epochs)):
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
            val_loss, all_predictions, all_targets, _ = evaluate_model(model, test_loader, criterion)
            val_losses.append(val_loss)
            test_acc.append(np.mean(np.array(all_predictions) == np.array(all_targets)))

            if test_acc[-1] > best_test_acc:
                best_test_acc = test_acc[-1]
                epochs_no_improve = 0

                # Save model state
                state_dict = model.state_dict()

                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor):
                        state_dict[k] = v.cpu()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= min_epochs:
                    print("Early stopping triggered!")
                    break

    if enable_profiling:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    train_acc = [float(i) for i in train_acc]
    test_acc = [float(i) for i in test_acc]

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

    # Load the best model
    model.load_state_dict(state_dict)

    return evaluate_model(model, test_loader, nn.CrossEntropyLoss())

def seed():
    torch.manual_seed(0)
    np.random.seed(0)

def setup_device():
    global device

    if torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


seglen_to_params = {
    1: { # 0.614
        "learning_rate": 0.00001,
        "batch_size": 16,
        "dropout": 0.35,
    },
    2: { # 0.619
        "learning_rate": 0.00001,
        "batch_size": 16,
        "dropout": 0.45,
    },
    3: { # 0.629, batch_size=8
        "learning_rate": 0.00001,
        "batch_size": 32,
        "dropout": 0.4,
    },
    5: { # 0.567
        "learning_rate": 0.00001,
        "batch_size": 8,
        "dropout": 0.4,
    },
    10: { # 0.622
        "learning_rate": 0.00001,
        "batch_size": 4,
        "dropout": 0.4,
    },
    15: { # 0.534
        "learning_rate": 0.00001,
        "batch_size": 16,
        "dropout": 0.4,
    },
    30: { # 0.589
        "learning_rate": 0.00001,
        "batch_size": 16,
        "dropout": 0.4,
    },
}

device = None
seglen = 3
use_gpu = True
min_epochs = 30

# val_splits = [[
#     8, 9, 10,  # Low DASPS
#     1, 2, 3, 4, 5, 6, 7, # High DASPS
#     *range(101, 106), # Low SAD
#     *range(401, 408)  # High SAD
# ]]

all_subj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                119, 120, 121, 401, 402, 403, 404, 405, 406,
                407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421
]

val_splits = [[i] for i in all_subj_ids]

def leave_subjects_out_cv(model, *, test_subj_ids, labeling_scheme, dataset_builder, seglen, params, max_epochs=100, min_epochs=min_epochs):
    train, test = dataset_builder.build_deep_datasets_train_test(
        seglen=seglen, insert_ch_dim=False, test_subj_ids=test_subj_ids, device=device, oversample=True)
    
    num_classes = labeling_scheme.get_num_classes()
    model = EEGNet(num_classes=num_classes, dropout=params["dropout"])
    model.to(device)

    return train_model(
        model, train, test, max_epochs=max_epochs, learning_rate=params["learning_rate"],
        enable_profiling=False, batch_size=params["batch_size"], min_epochs=min_epochs)
    
def gen_conf_matrix(all_targets, all_predictions, labeling_scheme: LabelingScheme):
    uniq_labels = np.unique(all_targets)
    uniq_labels_names = [labeling_scheme.get_label_name(i) for i in uniq_labels]
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=uniq_labels_names, yticklabels=uniq_labels_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    setup_device()

    # Build dataset
    labeling_scheme = LabelingScheme(DaspsLabeling.HAM, merge_control=False)
    builder = DatasetBuilder(labeling_scheme)

    params = seglen_to_params.get(seglen)

    if params is None:
        raise ValueError(f"No parameters defined for seglen: {seglen}")
    
    test_losses = []
    all_predictions = []
    all_targets = []
    all_accuracies = {}

    for test_subjs in val_splits:
        seed()

        _test_loss, _all_predictions, _all_targets, _all_accuracies = leave_subjects_out_cv(EEGNet, test_subj_ids=test_subjs, labeling_scheme=labeling_scheme,
            dataset_builder=builder, seglen=seglen, params=params, max_epochs=100)
        
        test_losses.append(_test_loss)
        all_predictions.extend(_all_predictions)
        all_targets.extend(_all_targets)

        for label in set(_all_accuracies.keys()):
            if label not in all_accuracies:
                all_accuracies[label] = []
            all_accuracies[label].append(_all_accuracies[label])

    # Statistics
    total_test_acc = np.mean(np.array(all_predictions) == np.array(all_targets))
    total_test_loss = np.mean(test_losses)
    avg_all_accuracies = {k: np.mean(v) for k, v in all_accuracies.items()}
    print(f"Total Test Loss: {total_test_loss}")
    print(f"Total Test Accuracy: {total_test_acc}")

    # Conf matrix
    gen_conf_matrix(all_targets, all_predictions, labeling_scheme)

    # Print label accuracies
    table = tabulate(all_accuracies.items(), headers=['Label', 'Accuracy'])
    print(table)
