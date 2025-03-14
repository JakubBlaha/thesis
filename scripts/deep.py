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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import csv
from datetime import datetime
import json

from models.cnn import EEGNet
from models.lstm import EEG_LSTMClassifier

import random

TEST_RUN = False

# Define LSTM parameters globally
lstm_enhanced = True

# Global control variables
show_plots = False
merge_control = True
oversample = True
mode = "both"

device = None
use_gpu = True

lstm_params = {
    "input_size": 14,
    "hidden_sizes": [45, 30],
    "bidirectional": lstm_enhanced,
    "use_attention": lstm_enhanced
}

all_subj_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 117, 118, 119, 120, 121, 401, 402, 403, 404, 405, 406, 407,
    408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421]

random.seed(42)
shuffled_ids = all_subj_ids.copy()
random.shuffle(shuffled_ids)

n_in_split = 1
val_splits = [
    shuffled_ids[i: i + n_in_split]
    for i in range(0, len(shuffled_ids),
                   n_in_split)]


if TEST_RUN:
    lstm_params['hidden_sizes'] = [1]


def compile_model(
        model, learning_rate=0.001, class_weights=None, l1_lambda=0.0):
    if class_weights is not None:
        class_weights = torch.tensor(
            class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return criterion, optimizer


def evaluate_model(model, test_loader, criterion):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            scores = model.forward(data)
            _, predictions = scores.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_predictions, all_targets


def train_model(
        model, train_dataset, test_dataset, *, max_epochs=100,
        learning_rate=0.001, batch_size=32, enable_profiling=False, patience=5,
        min_epochs=30, class_weights=None, l1_lambda=0.0):
    print("Train samples: ", len(train_dataset))
    print("Test samples: ", len(test_dataset))

    criterion, optimizer = compile_model(model, learning_rate, class_weights)

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

    epochs_no_improve = 0
    best_state_dict = None

    best_val_loss = float('inf')  # Initialize with a very high value
    best_val_acc = 0.0

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

                # L1 regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm

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
            all_predictions, all_targets = evaluate_model(
                model, test_loader, criterion)
            # Placeholder since we no longer calculate loss
            val_losses.append(0)
            test_acc.append(
                np.mean(np.array(all_predictions) == np.array(all_targets)))

            current_val_accuracy = test_acc[-1]

            if current_val_accuracy > best_val_acc and epoch >= min_epochs:
                best_val_acc = current_val_accuracy
                epochs_no_improve = 0

                # Save model state
                best_state_dict = model.state_dict()

                for k, v in best_state_dict.items():
                    if isinstance(v, torch.Tensor):
                        best_state_dict[k] = v.cpu()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= min_epochs:
                    print("Early stopping triggered!")
                    break

    if enable_profiling:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    train_acc = [float(i) for i in train_acc]
    test_acc = [float(i) for i in test_acc]

    plot_training_results(train_losses, val_losses, train_acc, test_acc)

    # Load the best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    all_predictions, all_targets = evaluate_model(
        model, test_loader, nn.CrossEntropyLoss())
    return (all_predictions, all_targets), test_acc


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


def plot_training_results(train_losses, val_losses, train_acc, test_acc):
    if not show_plots:
        return

    # Plotting with seaborn
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    configs = [
        (axs[0], {"Train Loss": train_losses, "Validation Loss": val_losses}, "Loss"),
        (axs[1], {"Train Accuracy": train_acc, "Test Accuracy": test_acc}, "Accuracy")
    ]
    for index, (ax, data_dict, title) in enumerate(configs):
        for label, values in data_dict.items():
            sns.lineplot(x=range(len(values)), y=values, ax=ax, label=label)

        if index == 1:
            ax.set(ylim=(0, 1))

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xlabel("Epochs")

    plt.show()


# Model-specific configurations
model_configs = {
    "cnn": {
        "learning_rate": 0.000005,
        "batch_size": 16,
        "dropout": 0.4,
        "class_weights": None,  # [1, 1, 1.3, 1]
        "l1_lambda": 0.0000,
        "min_epochs": 12,
        "max_epochs": 13
    },
    "lstm": {
        "learning_rate": 0.0005,
        "batch_size": 32,
        "dropout": 0.4,
        "class_weights": None,
        "l1_lambda": 0.0000,
        "min_epochs": 7,
        "max_epochs": 8
    }
}


def leave_subjects_out_cv(
        *, test_subj_ids, labeling_scheme,
        dataset_builder: DatasetBuilder, model_type):
    print("Test subjects: ", test_subj_ids)

    # Get current model config
    config = model_configs[model_type]

    train, test = dataset_builder.build_deep_datasets_train_test(
        insert_ch_dim=False, test_subj_ids=test_subj_ids, device=device)

    if len(test) == 0:
        return None

    num_classes = len(dataset_builder.last_int_to_label.keys())

    # Select model based on model_type
    if model_type == "cnn":
        model = EEGNet(num_classes=num_classes, dropout=config["dropout"])
    elif model_type == "lstm":
        model = EEG_LSTMClassifier(num_classes=num_classes,
                                   dropout=config["dropout"],
                                   **lstm_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)

    return train_model(
        model, train, test,
        max_epochs=config["max_epochs"],
        learning_rate=config["learning_rate"],
        enable_profiling=False,
        batch_size=config["batch_size"],
        min_epochs=config["min_epochs"],
        class_weights=config["class_weights"],
        l1_lambda=config["l1_lambda"])


def gen_conf_matrix(all_targets, all_predictions, int_to_label: dict):
    if not show_plots:
        return

    conf_matrix = confusion_matrix(all_targets, all_predictions)
    uniq_labels_names = [int_to_label[i] for i in sorted(int_to_label.keys())]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=uniq_labels_names, yticklabels=uniq_labels_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def save_results_to_csv(
        *, mean_accuracy, mean_test_loss=None, max_best_epoch=None,
        mean_best_epoch=None, model_config=None,
        seglen_value=None, model_type=None):
    # Create directory if it doesn't exist
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'results')
    os.makedirs(result_dir, exist_ok=True)

    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"deep_{timestamp}.csv"
    csv_filepath = os.path.join(result_dir, csv_filename)

    # Prepare row data
    row_data = {
        'timestamp': timestamp, 'mean_accuracy': mean_accuracy,
        'mean_test_loss': mean_test_loss, 'max_best_epoch': max_best_epoch,
        'mean_best_epoch': mean_best_epoch, 'seglen': seglen_value,
        'merge_control': merge_control, 'oversample': oversample,
        'n_in_split': n_in_split, 'model_type': model_type,
    }

    # Add model config values
    if model_config:
        for key, value in model_config.items():
            row_data[f"model_config_{key}"] = json.dumps(
                value) if isinstance(value, (list, dict, tuple)) else value

    # Add LSTM param values
    for key, value in lstm_params.items():
        row_data[f"lstm_{key}"] = json.dumps(value) if isinstance(
            value, (list, dict, tuple)) else value

    # Write to CSV
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
        writer.writeheader()
        writer.writerow(row_data)

    print(f"Results saved to {csv_filepath}")


def run_deep_learning(seglen: int, model_type_param: str):
    """
    Run the deep learning process with the specified parameters.

    Args:
        seglen_param: Segment length in seconds
        model_type_param: Model type ('lstm' or 'cnn')
    """
    setup_device()

    print(f"\n{'='*50}")
    print(
        f"Starting deep learning with seglen = {seglen}s, model = {model_type_param}")
    print(f"{'='*50}\n")

    # Build dataset
    labeling_scheme = LabelingScheme(
        DaspsLabeling.HAM, merge_control=merge_control)
    builder = DatasetBuilder(
        labeling_scheme, seglen=seglen, mode=mode, oversample=oversample)

    all_predictions = []
    all_targets = []
    best_epochs = []

    config = model_configs[model_type_param]  # Get current model config

    total_splits = len(val_splits)
    for split_idx, test_subjs in enumerate(val_splits):
        seed()

        ret = leave_subjects_out_cv(
            test_subj_ids=test_subjs,
            labeling_scheme=labeling_scheme,
            dataset_builder=builder,
            model_type=model_type_param)

        if ret is None:
            continue

        (_all_predictions, _all_targets), _test_accuracies = ret

        all_predictions.extend(_all_predictions)
        all_targets.extend(_all_targets)

        _best_epoch = np.argmax(
            _test_accuracies[config["min_epochs"]:]) + config["min_epochs"]
        _best_accuracy = _test_accuracies[_best_epoch]

        best_epochs.append(_best_epoch)

        print(f"Best epoch: {_best_epoch}")
        print("Best accuracy: ", _best_accuracy)
        print(
            f"Progress: {split_idx+1}/{total_splits} folds completed for seglen={seglen}s")
        print("----------------")

        if TEST_RUN:
            break

    # Statistics
    total_test_acc = np.mean(np.array(all_predictions)
                             == np.array(all_targets))

    # Calculate mean and max of best epochs
    mean_best_epoch = np.mean(best_epochs)
    max_best_epoch = np.max(best_epochs)
    print(f"Mean Best Epoch: {mean_best_epoch}")
    print(f"Max Best Epoch: {max_best_epoch}")

    # Save results to CSV
    save_results_to_csv(
        mean_accuracy=total_test_acc,
        max_best_epoch=max_best_epoch,
        mean_best_epoch=mean_best_epoch,
        model_config=config,
        seglen_value=seglen,
        model_type=model_type_param
    )

    # Conf matrix
    gen_conf_matrix(all_targets, all_predictions, builder.last_int_to_label)

    # Save predictions to CSV
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'results')
    os.makedirs(result_dir, exist_ok=True)

    # Use the same timestamp format as in save_results_to_csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_filename = f"deep_{timestamp}_predictions.csv"
    predictions_filepath = os.path.join(result_dir, predictions_filename)

    # Create a dataframe with predictions and actual labels
    with open(predictions_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predicted', 'actual'])
        for pred, actual in zip(all_predictions, all_targets):
            pred_ = builder.last_int_to_label[pred]
            actual_ = builder.last_int_to_label[actual]
            writer.writerow([pred_, actual_])

    print(f"Predictions saved to {predictions_filepath}")

    print("\nScript execution finished.")


if __name__ == "__main__":
    # seglens = [1, 2, 3, 5]
    # seglens = [3, 5, 10, 15, 30]
    seglens = [30]

    for seglen in seglens:
        run_deep_learning(seglen=seglen, model_type_param="cnn")
