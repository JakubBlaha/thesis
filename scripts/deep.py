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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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

lstm_params = {
    "input_size": 14,
    "hidden_sizes": [45, 30],
    "bidirectional": lstm_enhanced,
    "use_attention": lstm_enhanced
}

# Global variables for parameters
merge_control = True
oversample = True
mode = "both"

device = None
use_gpu = True

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
    label_precisions = {}
    label_recalls = {}
    label_f1scores = {}

    # Get unique labels in targets
    unique_labels = set(all_targets)

    # Calculate metrics per label
    for label in unique_labels:
        # Calculate accuracy
        label_indices = [i for i, target in enumerate(
            all_targets) if target == label]
        label_predictions = [all_predictions[i] for i in label_indices]
        correct_predictions = sum(
            1 for pred in label_predictions if pred == label)
        accuracy = correct_predictions / len(
            label_predictions) if len(label_predictions) > 0 else 0
        label_accuracies[label] = accuracy

        # Calculate precision, recall, and f1 using binary classification approach for each class
        y_true = [1 if t == label else 0 for t in all_targets]
        y_pred = [1 if p == label else 0 for p in all_predictions]

        # Handle potential division by zero
        label_precisions[label] = precision_score(
            y_true, y_pred, zero_division=0)
        label_recalls[label] = recall_score(y_true, y_pred, zero_division=0)
        label_f1scores[label] = f1_score(y_true, y_pred, zero_division=0)

    # Calculate macro averages
    macro_precision = sum(label_precisions.values(
    )) / len(label_precisions) if label_precisions else 0
    macro_recall = sum(label_recalls.values()
                       ) / len(label_recalls) if label_recalls else 0
    macro_f1 = sum(label_f1scores.values()
                   ) / len(label_f1scores) if label_f1scores else 0

    return (avg_loss, all_predictions, all_targets, label_accuracies,
            label_precisions, label_recalls, label_f1scores,
            macro_precision, macro_recall, macro_f1)


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
            val_loss, all_predictions, all_targets, _, _, _, _, _, _, _ = evaluate_model(
                model, test_loader, criterion)
            val_losses.append(val_loss)
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

    return evaluate_model(model, test_loader, nn.CrossEntropyLoss()), test_acc


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
        *, mean_accuracy, mean_test_loss, max_best_epoch, mean_best_epoch,
        model_config, int_to_label, seglen_value,
        model_type, macro_precision, macro_recall, macro_f1):
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
        'macro_precision': macro_precision, 'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'int_to_label': json.dumps(
            {str(k): v for k, v in int_to_label.items()})}

    # Add model config values
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

    test_losses = []
    all_predictions = []
    all_targets = []
    all_accuracies = {}
    all_precisions = {}
    all_recalls = {}
    all_f1scores = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    best_epochs = []
    group_test_accuracies = {}

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

        (_test_loss, _all_predictions, _all_targets,
         _label_accuracies, _label_precisions, _label_recalls, _label_f1scores,
         _macro_precision, _macro_recall, _macro_f1), _test_accuracies = ret

        test_losses.append(_test_loss)
        all_predictions.extend(_all_predictions)
        all_targets.extend(_all_targets)
        macro_precisions.append(_macro_precision)
        macro_recalls.append(_macro_recall)
        macro_f1s.append(_macro_f1)

        for label in set(_label_accuracies.keys()):
            if label not in all_accuracies:
                all_accuracies[label] = []
                all_precisions[label] = []
                all_recalls[label] = []
                all_f1scores[label] = []

            all_accuracies[label].append(_label_accuracies[label])
            all_precisions[label].append(_label_precisions.get(label, 0))
            all_recalls[label].append(_label_recalls.get(label, 0))
            all_f1scores[label].append(_label_f1scores.get(label, 0))

        _best_epoch = np.argmax(
            _test_accuracies[config["min_epochs"]:]) + config["min_epochs"]
        _best_accuracy = _test_accuracies[_best_epoch]

        best_epochs.append(_best_epoch)

        print(f"Best epoch: {_best_epoch}")
        print("Best accuracy: ", _best_accuracy)
        print("Macro precision: ", _macro_precision)
        print("Macro recall: ", _macro_recall)
        print("Macro F1: ", _macro_f1)
        print(
            f"Progress: {split_idx+1}/{total_splits} folds completed for seglen={seglen}s")
        print("----------------")

        group_test_accuracies[tuple(test_subjs)] = _best_accuracy

        if TEST_RUN:
            break

    # Statistics
    total_test_acc = np.mean(np.array(all_predictions)
                             == np.array(all_targets))
    total_test_loss = np.mean(test_losses)
    avg_all_accuracies = {k: np.mean(v) for k, v in all_accuracies.items()}
    avg_all_precisions = {k: np.mean(v) for k, v in all_precisions.items()}
    avg_all_recalls = {k: np.mean(v) for k, v in all_recalls.items()}
    avg_all_f1scores = {k: np.mean(v) for k, v in all_f1scores.items()}

    # Calculate average macro metrics across all splits
    avg_macro_precision = np.mean(macro_precisions)
    avg_macro_recall = np.mean(macro_recalls)
    avg_macro_f1 = np.mean(macro_f1s)

    print(f"Mean Test Loss: {total_test_loss}")
    print(f"Mean Test Accuracy: {total_test_acc}")
    print(f"Mean Macro Precision: {avg_macro_precision}")
    print(f"Mean Macro Recall: {avg_macro_recall}")
    print(f"Mean Macro F1: {avg_macro_f1}")

    # Calculate mean and max of best epochs
    mean_best_epoch = np.mean(best_epochs)
    max_best_epoch = np.max(best_epochs)
    print(f"Mean Best Epoch: {mean_best_epoch}")
    print(f"Max Best Epoch: {max_best_epoch}")

    # Save results to CSV
    save_results_to_csv(
        mean_accuracy=total_test_acc,
        mean_test_loss=total_test_loss,
        max_best_epoch=max_best_epoch,
        mean_best_epoch=mean_best_epoch,
        model_config=config,
        int_to_label=builder.last_int_to_label,
        seglen_value=seglen,
        model_type=model_type_param,
        macro_precision=avg_macro_precision,
        macro_recall=avg_macro_recall,
        macro_f1=avg_macro_f1
    )

    # Conf matrix
    gen_conf_matrix(all_targets, all_predictions, builder.last_int_to_label)

    # Print metrics per class
    print("\nClass Metrics:")
    metrics = []

    for label in sorted(avg_all_accuracies.keys()):
        class_name = builder.last_int_to_label[label]
        metrics.append([
            class_name,
            avg_all_accuracies.get(label, 0),
            avg_all_precisions.get(label, 0),
            avg_all_recalls.get(label, 0),
            avg_all_f1scores.get(label, 0)
        ])

    print(tabulate(
        metrics,
        headers=['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score']))

    print("Total splits: ", len(val_splits))
    print("Script execution finished.")


if __name__ == "__main__":
    # seglens = [1, 2, 3, 5]
    # seglens = [3, 5, 10, 15, 30]
    seglens = [30]

    for seglen in seglens:
        run_deep_learning(seglen=seglen, model_type_param="cnn")
