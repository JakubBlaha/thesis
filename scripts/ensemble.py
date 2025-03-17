import os
import numpy as np
import pandas as pd
import datetime
import argparse

from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Assume these are defined in your project (or replace with your own dataset loading)
from utils import DatasetBuilder, LabelingScheme, DaspsLabeling


def train_model(*, seglen, mode, domains, strategy, oversample=True):
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
            kernel='rbf', C=10, gamma=0.1,
            probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(
            n_neighbors=5, weights='uniform')),
        ('mlp',
         MLPClassifier(
             hidden_layer_sizes=(20,),
             activation='relu', solver='adam', alpha=1e-7,
             learning_rate='constant', random_state=42, early_stopping=True)),
        ('rf',
         RandomForestClassifier(
             n_estimators=400, max_depth=20, random_state=42))]

    # Choose ensemble type based on selected strategy
    if strategy == "voting":
        ensemble_classifier = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
    else:  # stacking strategy
        ensemble_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=10
        )

    # Create a pipeline with scaling, feature selection, and the classifier.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sel', SelectKBest(score_func=f_classif)),
        ('classif', ensemble_classifier)
    ])

    # Grid search to find the best number of features.
    param_grid = {
        'sel__k': [20, 40, 60, 80, 100, 'all']
        # 'sel__k': [10]
    }
    grid = GridSearchCV(pipeline, param_grid,
                        cv=cv_strategy, n_jobs=-1, verbose=10)
    grid.fit(features, labels, groups=groups)
    best_pipeline = grid.best_estimator_
    best_k = grid.best_params_['sel__k']
    print("Best number of features:", best_k)

    # Obtain cross-validated predictions.
    predictions = cross_val_predict(
        best_pipeline, features, labels,
        cv=cv_strategy, groups=groups, n_jobs=-1
    )

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

    # Create the results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save the file in the results directory
    filename = os.path.abspath(os.path.join(
        results_dir,
        f"{classifier_name}_seglen_{seglen}_k_{best_k}_predictions_{timestamp}.csv"
    ))
    df_results.to_csv(filename, index=False)
    print("Saved predictions to:", filename)


# Example usage:
if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Train an ensemble classifier model.')
    parser.add_argument(
        '--strategy', '-s', type=str, required=True,
        choices=['voting', 'stacking'],
        help='Ensemble strategy: "voting" for VotingClassifier or "stacking" for StackingClassifier')
    parser.add_argument('--seglen', type=int, default=15,
                        help='Segment length (default: 15)')
    parser.add_argument('--mode', type=str, default="both",
                        help='Mode (default: "both")')
    parser.add_argument('--domains', type=str, nargs='+',
                        default=["rel_pow", "conn", "ai", "time", "abs_pow"],
                        help='Domains to include')
    parser.add_argument('--oversample', action='store_true',
                        help='Whether to oversample the dataset')
    parser.add_argument(
        '--no-oversample', action='store_false', dest='oversample',
        help='Disable oversampling')
    parser.set_defaults(oversample=True)

    args = parser.parse_args()

    # Train a single ensemble model with the specified strategy
    print(
        f"\nTraining ensemble classifier with strategy: {args.strategy}, seglen: {args.seglen}")
    train_model(
        seglen=args.seglen,
        mode=args.mode,
        domains=args.domains,
        strategy=args.strategy,
        oversample=args.oversample)
