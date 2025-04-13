import os
import datetime
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectKBest, f_classif
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd

from utils import DatasetBuilder, LabelingScheme, DaspsLabeling
from tabulate import tabulate

verbosity = 10

script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
results_dir = os.path.join(data_dir, 'results')

os.makedirs(results_dir, exist_ok=True)

# This was used for hyperparameter tuning.
# The hypertparameter ranges were edited here in the source code.
# These hyperparameters are not an accurate representation of what space
# was actually searched. The hyperparameters below are only a subset
# of the hyperparameters that were actually searched.

GRID = {
    "svm-rbf": {
        "classif": svm.SVC,

        # Three class
        # "params": {
        #     "classif__C": [10**i for i in range(-3, 5)],
        #     "classif__gamma": [10**i for i in range(-5, 0)],
        #     "classif__kernel": ['rbf'],
        #     "classif__max_iter": [10_000],
        #     "sel__k": [5, 8, 10, 20, 30, 40, 60, 100, "all"]
        # },

        # Binary SAD
        # "params": {
        #     "classif__C": [100],
        #     "classif__gamma": [0.01],
        #     "classif__kernel": ['rbf'],
        #     "classif__max_iter": [20_000],
        #     "sel__k": [5, 8, 10, 20, 30, 40, 60, 100, "all"]
        # }

        # Binary DASPS
        "params": {
            "classif__C": [10**i for i in range(-3, 5)],
            "classif__gamma": [10**i for i in range(-5, 0)],
            "classif__kernel": ['rbf'],
            "classif__max_iter": [10_000],
            "sel__k": [20],
            # "sel__k": [5, 8, 10, 20, 30, 40, 60, 100],
            "classif__class_weight": ['balanced'],
        },
    },
    # Linear kernel seems to work better without standard scaler
    "svm-lin": {
        "classif": svm.LinearSVC,
        "params": {
            "classif__C": [10**i for i in range(-3, 4)],
            "classif__max_iter": [5000],
            "sel__k": [5, 8, 10, 20, 30, 40, 60, "all"],
        }
    },
    # "svm-lin": {
    #     "classif": svm.LinearSVC,
    #     "params": {
    #         "classif__C": [10**i for i in range(-3, 4)],
    #         "sel__k": [60]
    #     }
    # },
    "svm-poly": {
        "classif": svm.SVC,
        "params": {
            "classif__C": [10**i for i in range(-3, 2)],
            "classif__gamma": [10**i for i in range(-4, 0)],
            "classif__kernel": ['poly'],
            "classif__degree": [2, 3, 4],
            "classif__max_iter": [5000],
            "sel__k": [20, 30, 40, 60, "all"]
        }
    },
    "rf": {
        "classif": RandomForestClassifier,

        # "params": {
        #     "classif__n_estimators": [300, 400, 450],
        #     "classif__max_depth": [5, 6, 8, 10, 15, 20],
        #     # "min_samples_split": [2, 5, 10],
        #     # "min_samples_leaf": [1, 2, 4],
        #     # "max_features": ['auto', 'sqrt', 'log2'],
        #     # "sel__k": [5, 10, 20, 40, 60, "all"],
        #     "sel__k": [5, 10, 20, 40, 60, "all"]
        # },

        # DASPS
        "params": {
            "classif__n_estimators": [300],
            "classif__max_depth": [8],
            "classif__class_weight": ['balanced'],
            # "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [1, 2, 4],
            # "max_features": ['auto', 'sqrt', 'log2'],
            # "sel__k": [5, 10, 20, 40, 60, "all"],
            "sel__k": [5, 10, 20, 40, 60, "all"]
        },
    },
    # "nb": {
    #     "classif": GaussianNB,
    #     "params": {
    #         "classif__var_smoothing": [10**i for i in range(-9, 0)],
    #         "sel__k": [5, 8, 10, 20, 30, 40, 60]
    #     }
    # },
    "knn": {
        "classif": KNeighborsClassifier,
        "params": {
            "classif__n_neighbors": [3, 5, 8, 10, 15, 20, 30],
            "classif__weights": ['uniform', 'distance'],
            "sel__k": [1, 2, 3, 4, 5, 8, 10, 20, 30, 40, 60],
            # "sel__k": [60],
        }
    },
    "mlp": {
        "classif": MLPClassifier,
        "params": {
            "classif__hidden_layer_sizes": [(30,), (20,), (10,)],
            # "classif__hidden_layer_sizes": [(20,)],
            "classif__activation": ['relu'],
            "classif__solver": ['adam'],
            "classif__alpha": [10**i for i in range(-9, -5)],
            "classif__learning_rate": ['constant'],
            "classif__max_iter": [2000],
            "sel__k": [20, 40, 60, "all"],
            # "sel__k": [60]
        },
    },
    # "lda": {
    #     "classif": LinearDiscriminantAnalysis,
    #     "params": {
    #         # "classif__solver": ['svd', 'lsqr', 'eigen'],
    #         "classif__solver": ['lsqr'],
    #         # "classif__shrinkage": [None, 'auto'] + [i/10.0 for i in range(1, 10)],
    #         "classif__shrinkage": ['auto'],
    #         # "classif__tol": [10**i for i in range(-9, 0)],
    #         # "classif__n_components": [1, 2, 3, 5, None],
    #         # "sel__k": [5, 8, 10, 20, 30, 40, 60, "all"],
    #         "sel__k": [20, 40, 60, "all"],
    #     }
    # },
}


def train_model(
        *, classif, seglen, mode, domains, dasps_labeling_scheme="ham",
        oversample=True, cv='logo'):
    """
    Train a single machine learning model with the specified configuration.

    This function builds a dataset, performs feature selection, trains the model using
    grid search for hyperparameter tuning, and evaluates model performance using
    cross-validation.

    Parameters
    ----------
    classif : str
        Name of the classifier to use (must be a key in the GRID dictionary).
    seglen : int
        Segment length in seconds to use for feature extraction.
    mode : str
        Mode of data processing (e.g., 'concat', 'mean').
    domains : list of str
        List of data domains to include in the dataset.
    dasps_labeling_scheme : str, optional
        The labeling scheme to use, either 'ham' or 'sam', default is 'ham'.
    oversample : bool, optional
        Whether to use oversampling to balance class distribution, default is True.
    cv : str, optional
        Cross-validation strategy, either 'logo' (Leave One Group Out) or 'skf' 
        (Stratified K-Fold), default is 'logo'.

    Returns
    -------
    float
        Mean accuracy score from cross-validation.
    float
        Mean F1 score (macro-averaged).
    float
        Mean precision score (macro-averaged).
    float
        Mean recall score (macro-averaged).
    str
        String representation of the best hyperparameters.
    int
        Number of selected features.
    str
        Comma-separated list of selected feature names.
    """
    if dasps_labeling_scheme == "ham":
        _labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    elif dasps_labeling_scheme == "sam":
        _labeling_scheme = LabelingScheme(DaspsLabeling.SAM)

    builder = DatasetBuilder(_labeling_scheme, seglen,
                             mode, oversample=oversample)
    features, labels, groups, df = builder.build_dataset_feats_labels_groups_df(
        domains)

    n_feats = features.shape[1]

    print("Number of features:", n_feats)

    print("Label distribution:")
    print(pd.Series(labels).value_counts())

    int_to_label = builder.last_int_to_label

    print("Label mapping:")
    print(int_to_label)

    if cv == 'logo':
        _cv = LeaveOneGroupOut()
    elif cv == 'skf':
        _cv = StratifiedKFold(n_splits=10)

    param_grid = GRID[classif]["params"]

    # Make sure that the number of features in the grid is never
    # greater than the number of available features
    if "sel__k" in param_grid:
        _corrected_feat_selection_grid = []

        for i in param_grid["sel__k"]:
            if type(i) == int:
                if i <= n_feats:
                    _corrected_feat_selection_grid.append(i)
            else:
                _corrected_feat_selection_grid.append(i)

        param_grid["sel__k"] = _corrected_feat_selection_grid
        # param_grid["sel__k"] = [60]

    # Create pipeline with feature selection and SVM
    pipeline = Pipeline([
        # ('scaler', MinMaxScaler((0, 1))),
        ('scaler', StandardScaler()),
        ('sel', SelectKBest(score_func=f_classif)),
        # ('sel', SelectFpr(score_func=f_classif, alpha=0.05)),
        ('classif', GRID[classif]["classif"]())
    ])

    search = GridSearchCV(pipeline, param_grid,
                          n_jobs=-1, verbose=verbosity, cv=_cv)

    _search_kw = {}

    if cv == 'logo':
        _search_kw['groups'] = groups

    search.fit(features, labels, **_search_kw)

    best_estimator = search.best_estimator_

    scores = cross_val_score(
        best_estimator, features, labels, groups=groups, cv=_cv,
        verbose=verbosity, n_jobs=-1)

    # Get the unique class labels to ensure consistent confusion matrix dimensions
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Calculate additional metrics using cross-validation
    f1_scores = []
    precision_scores = []
    recall_scores = []
    conf_matrices = []

    for train_idx, test_idx in _cv.split(
            features, labels, groups if cv == 'logo' else None):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        precision_scores.append(precision_score(
            y_test, y_pred, average='macro'))
        recall_scores.append(
            recall_score(
                y_test, y_pred, average='macro',
                zero_division=0))

        # Create confusion matrix with consistent labels
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        conf_matrices.append(cm)

    # Output

    print("Searched parameters grid:")
    if param_grid:  # Only try to tabulate if there are parameters
        param_table = {
            "Parameter": list(param_grid.keys()),
            "Values": [str(values) for values in param_grid.values()]
        }
        print(tabulate(param_table, headers="keys",
              tablefmt="pretty", maxcolwidths=30))
    else:
        print("No parameters in grid.")

    # Print selected features
    selected_features = search.best_estimator_.named_steps['sel']
    selected_mask = selected_features.get_support()
    num_selected_features = sum(selected_mask)
    print("Number of selected features:", num_selected_features)

    # Selected features
    selected_features_names = df.columns[selected_mask]
    selected_features_table = {
        "Index": range(len(selected_features_names)),
        "Feature Name": selected_features_names
    }

    # Create string of selected feature names for storage
    selected_features_str = ", ".join(selected_features_names)

    print("Selected features:")
    print(tabulate(selected_features_table, headers="keys", tablefmt="pretty"))

    print("Best parameters:")
    best_params = {
        "Parameter": list(search.best_params_.keys()),
        "Value": list(search.best_params_.values())
    }

    print(tabulate(best_params, headers="keys", tablefmt="pretty"))

    # Print scores as a table
    score_results = {
        "Metric":
        ["Mean accuracy", "Std accuracy", "Mean F1 (macro)",
         "Mean Precision (macro)", "Mean Recall (macro)"],
        "Value":
        [round(i, 2)
         for i
         in
         [scores.mean(),
          scores.std(),
          np.mean(f1_scores),
          np.mean(precision_scores),
          np.mean(recall_scores)]]}

    print(tabulate(score_results, headers="keys", tablefmt="pretty"))

    config_table = {
        "Parameter": ["Classifier", "Mode", "CV", "Segment length", "Oversample", "Domains", "Labeling scheme"],
        "Value": [classif, mode, cv, seglen, oversample, ", ".join(domains), dasps_labeling_scheme]
    }

    print("Configuration:")
    print(
        tabulate(
            config_table, headers="keys", tablefmt="pretty",
            maxcolwidths=30))

    # Print confusion matrix with proportions (0-1)
    # First sum all matrices instead of averaging
    sum_conf_matrix = np.sum(conf_matrices, axis=0)

    # Normalize by row (convert to proportions 0-1)
    row_sums = sum_conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = np.zeros_like(sum_conf_matrix, dtype=float)
    # Avoid division by zero
    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            # Now proportions, not percentages
            norm_conf_matrix[i] = sum_conf_matrix[i] / row_sums[i]

    print("\nConfusion Matrix (proportions):")
    # Round to 2 decimal places for display
    norm_conf_matrix_rounded = np.round(norm_conf_matrix, 2)
    cm_df = pd.DataFrame(norm_conf_matrix_rounded,
                         index=[f'True {int_to_label[i]} '
                                for i in unique_labels],
                         columns=[f'Pred {int_to_label[i]} '
                                  for i in unique_labels])
    print(tabulate(cm_df, headers="keys", tablefmt="pretty"))

    # Also show the raw counts
    print("\nConfusion Matrix (counts):")
    cm_counts_df = pd.DataFrame(sum_conf_matrix,
                                index=[f'True {int_to_label[i]} '
                                       for i in unique_labels],
                                columns=[f'Pred {int_to_label[i]} '
                                         for i in unique_labels])
    print(tabulate(cm_counts_df, headers="keys", tablefmt="pretty"))

    # Convert best parameters to a string for storage
    best_params_str = ", ".join(
        [f"{k}={v} "for k, v in search.best_params_.items()])

    return scores.mean(), np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores), best_params_str, num_selected_features, selected_features_str


def train_models(
        *, seglens, mode, domains, dasps_labeling_scheme, oversample, cv,
        classifiers):
    """
    Train multiple models with different configurations and save results to a CSV file.

    This function iterates through combinations of classifiers and segment lengths,
    trains models for each combination, and compiles the results into a DataFrame
    which is then saved as a CSV file.

    Parameters
    ----------
    seglens : list of int
        List of segment lengths to use for training.
    mode : str
        Mode of data processing (e.g., 'concat', 'mean').
    domains : list of str
        List of data domains to include in the dataset.
    dasps_labeling_scheme : str
        The labeling scheme to use, either 'ham' or 'sam'.
    oversample : bool
        Whether to use oversampling to balance class distribution.
    cv : str
        Cross-validation strategy, either 'logo' (Leave One Group Out) or 'skf' 
        (Stratified K-Fold).
    classifiers : list of str
        List of classifier names to train (must be keys in the GRID dictionary).

    Returns
    -------
    None
        Results are saved to a CSV file in the results directory.
    """
    results = []

    # Validate that all requested classifiers exist in GRID
    valid_classifiers = []
    for clf in classifiers:
        if clf in GRID:
            valid_classifiers.append(clf)
        else:
            print(
                f"Warning: Classifier '{clf}' not found in GRID. Available classifiers: {list(GRID.keys())}")

    if not valid_classifiers:
        print(
            f"Error: None of the requested classifiers are available. Available classifiers: {list(GRID.keys())}")
        return

    for clf in valid_classifiers:
        for s in seglens:
            print(f"Training for classifier: {clf} with seglen: {s}")
            acc, f1, precision, recall, best_params, num_features, feature_names = train_model(
                classif=clf, seglen=s, mode=mode, domains=domains,
                dasps_labeling_scheme=dasps_labeling_scheme,
                oversample=oversample, cv=cv)
            results.append({
                'classifier': clf,
                'seglen': s,
                'mean_accuracy': acc,
                'macro_f1': f1,
                'macro_precision': precision,
                'macro_recall': recall,
                'best_params': best_params,
                'num_selected_features': num_features,
                'selected_features': feature_names
            })

    df_results = pd.DataFrame(results)

    # Round numeric columns to 3 decimal places
    numeric_columns = ['mean_accuracy', 'macro_f1',
                       'macro_precision', 'macro_recall']
    df_results[numeric_columns] = df_results[numeric_columns].round(3)

    # Generate timestamp and create a unique filename with all parameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seglens_str = '-'.join(map(str, seglens))
    domains_str = '-'.join(domains)
    oversample_str = "os" if oversample else "nos"

    filename = f"classif_{timestamp}_mode-{mode}_seglens-{seglens_str}_domains-{domains_str}_label-{dasps_labeling_scheme}_cv-{cv}_{oversample_str}.csv"
    csv_path = os.path.join(results_dir, filename)

    df_results.to_csv(csv_path, index=False)

    print("Results saved to:", csv_path)
