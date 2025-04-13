# Author: Jakub Bláha, xblaha36

"""
Ensemble learning module for physiological data classification.

This script implements ensemble learning strategies (voting and stacking) for classification
of physiological signals. It supports various machine learning classifiers as base estimators
and provides hyperparameter optimization through grid search.
"""

import os
import numpy as np
import pandas as pd
import datetime

from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from utils import DatasetBuilder, LabelingScheme, DaspsLabeling


"""
Dictionary of final classifiers for stacking ensemble with their parameter grids.

Keys are classifier names, and values are dictionaries containing:
- 'classifier': The sklearn classifier instance
- 'param_grid': Parameters to search during grid search optimization
"""
FINAL_CLASSIFIERS = {
    "logistic": {
        "classifier": LogisticRegression(random_state=42),
        "param_grid": {
            'classif__final_estimator__C': [0.01],
            'classif__final_estimator__penalty': ['l1'],
            'classif__final_estimator__solver': ['liblinear']
        }
    },
    "rf": {
        "classifier": RandomForestClassifier(random_state=42),
        "param_grid": {
            # 'classif__final_estimator__n_estimators': [10, 20, 30, 50, 100],
            'classif__final_estimator__n_estimators': [30],
            # 'classif__final_estimator__max_depth': [2, 3, 4],
            'classif__final_estimator__max_depth': [2],
            'classif__final_estimator__max_features': ['sqrt'],
        }
    },
    "mlp": {
        "classifier": MLPClassifier(random_state=42, early_stopping=True),
        "param_grid": {
            # 'classif__final_estimator__hidden_layer_sizes': [(10,), (20,), (30,), (10, 10)],
            'classif__final_estimator__hidden_layer_sizes': [(30,)],
            # 'classif__final_estimator__alpha': [1e-8, 1e-9, 1e-7],
            'classif__final_estimator__alpha': [1e-8],
        }
    },
    "gb": {
        "classifier": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            'classif__final_estimator__n_estimators': [200],
            'classif__final_estimator__learning_rate': [0.2, 0.5, 1],
            'classif__final_estimator__max_depth': [1, 2, 3]
        }
    }
}


def train_model(*, seglen, mode, domains, strategy,
                final_classifier="logistic", oversample=True, seed=42):
    """
    Train and evaluate an ensemble model with specified configuration.

    Builds a dataset, trains an ensemble classifier (voting or stacking), performs
    hyperparameter optimization, and evaluates the model using leave-one-group-out
    cross-validation. Results are saved to a CSV file.

    Parameters
    ----------
    seglen : int
        Segment length for data preprocessing.
    mode : str
        Dataset building mode.
    domains : list
        List of domains to include in dataset.
    strategy : str
        Ensemble strategy, either "voting" or "stacking".
    final_classifier : str, default="logistic"
        Final classifier type for stacking ("logistic", "rf", "mlp", or "gb").
        Ignored when strategy is "voting".
    oversample : bool, default=True
        Whether to apply oversampling to balance the dataset.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    None
        Results and model parameters are saved to a CSV file in '../data/results'.
    """
    # Set seed for reproducibility
    np.random.seed(seed)

    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)

    # Build the dataset: features, labels, groups, and a dataframe with feature names.
    builder = DatasetBuilder(labeling_scheme, seglen,
                             mode, oversample=oversample)
    features, labels, groups, df = builder.build_dataset_feats_labels_groups_df(
        domains)
    print("Number of features:", features.shape[1])

    # Use only LeaveOneGroupOut for cross-validation
    cv_strategy = LeaveOneGroupOut()

    # Create ensemble model directly here
    estimators = [
        ('svm', svm.SVC(
            kernel='rbf', C=10, gamma=0.1, probability=True,
            random_state=seed)),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='uniform')),
        ('mlp',
         MLPClassifier(
             hidden_layer_sizes=(20,),
             activation='relu', solver='adam', alpha=1e-7,
             learning_rate='constant', random_state=seed,
             early_stopping=True)),
        ('rf',
         RandomForestClassifier(
             n_estimators=400, max_depth=20, random_state=seed))]

    # Choose ensemble type based on selected strategy
    if strategy == "voting":
        ensemble_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        # Parameter grid for voting strategy
        param_grid = {
            'sel__k': [80]
        }
    else:  # stacking strategy
        # Get the selected final classifier
        if final_classifier not in FINAL_CLASSIFIERS:
            raise ValueError(f"Unknown final classifier: {final_classifier}. "
                             f"Available options: {', '.join(FINAL_CLASSIFIERS.keys())}")

        selected_classifier = FINAL_CLASSIFIERS[final_classifier]

        # Create a copy of the classifier with the provided seed
        final_estimator = selected_classifier["classifier"]
        if hasattr(final_estimator, 'random_state'):
            final_estimator.set_params(random_state=seed)

        ensemble_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )

        # Base parameter grid with common parameters
        param_grid = {
            # 'sel__k': [40, 60, 80]
            'sel__k': [40]
        }

        # Add classifier-specific parameters
        param_grid.update(selected_classifier["param_grid"])
        print(
            f"Using {final_classifier} as final classifier with parameter grid: {param_grid}")

    # Create a pipeline with scaling, feature selection, and the classifier.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sel', SelectKBest(score_func=f_classif)),
        ('classif', ensemble_classifier)
    ])

    # Grid search to find the best parameters
    grid = GridSearchCV(pipeline, param_grid,
                        cv=cv_strategy, n_jobs=-1, verbose=10)
    grid.fit(features, labels, groups=groups)
    best_pipeline = grid.best_estimator_
    best_params = grid.best_params_
    print("Best parameters:", best_params)
    best_k = best_params['sel__k']
    print("Best number of features:", best_k)

    # Obtain cross-validated predictions.
    predictions = cross_val_predict(
        best_pipeline, features, labels,
        cv=cv_strategy, groups=groups, n_jobs=-1
    )

    # Calculate and print the accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"Best model accuracy: {accuracy:.3f}")

    # Convert integer labels to string representations
    actual_labels_str = [
        builder.last_int_to_label[int(label)] for label in labels]
    predicted_labels_str = [
        builder.last_int_to_label[int(pred)] for pred in predictions]

    # Save actual and predicted labels to a CSV file in '../data/results' directory
    df_results = pd.DataFrame({
        "predicted": predicted_labels_str,
        "actual": actual_labels_str
    })
    timestamp = str(int(datetime.datetime.now().timestamp()))
    classifier_name = f"ensemble-{strategy}"
    if strategy == "stacking":
        classifier_name += f"-{final_classifier}"

    # Add accuracy to the parameter information
    best_params['accuracy'] = accuracy

    # Create the results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save the file in the results directory
    filename = os.path.abspath(os.path.join(
        results_dir,
        f"{classifier_name}_seglen_{seglen}_k_{best_k}_predictions_{timestamp}.csv"
    ))

    # First save the prediction results
    df_results.to_csv(filename, index=False)

    # Then append the parameters as additional rows
    with open(filename, 'a') as f:
        f.write("\n\n# Model Parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param},{value}\n")

    print("Saved predictions and parameters to:", filename)
