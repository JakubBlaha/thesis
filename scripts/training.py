# %%
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from utils import DatasetBuilder, LabelingScheme, DaspsLabeling
from tabulate import tabulate

GRID = {
    "svm-rbf": {
        "classif": svm.SVC,
        "params": {
            "classif__C": [10**i for i in range(-3, 5)],
            "classif__gamma": [10**i for i in range(-9, 0)],
            "classif__kernel": ['rbf'],
            "sel__k": [5, 8, 10, 20, 30, 40, 60, 100, "all"]
        }
    },
    # Linear kernel seems to work better without standard scaler
    "svm-lin": {
        "classif": svm.LinearSVC,
        "params": {
            "classif__C": [10**i for i in range(-3, 4)],
            "sel__k": [5, 8, 10, 20, 30, 40, 60, "all"]
        }
    },
    "svm-poly": {
        "classif": svm.SVC,
        "params": {
            "classif__C": [10**i for i in range(-3, 5)],
            "classif__gamma": [10**i for i in range(-9, 0)],
            "classif__kernel": ['poly'],
            "classif__degree": [2, 3, 4],
            "sel__k": [5, 8, 10, 20, 30, 40, 60, "all"]
        }
    },
    # Random forest works best with
    "rf": {
        "classif": RandomForestClassifier,
        "params": {
            "classif__n_estimators": [300],
            "classif__max_depth": [5, 6, 8, 10, 15, 20],
            # "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [1, 2, 4],
            # "max_features": ['auto', 'sqrt', 'log2'],
            "sel__k": [5, 8, 10, 20, 30, 40, 60]
        }
    },
    "nb-gauss": {
        "classif": GaussianNB,
        "params": {
            "classif__var_smoothing": [10**i for i in range(-9, 0)],
            "sel__k": [5, 8, 10, 20, 30, 40, 60]
        }
    },
    "knn": {
        "classif": KNeighborsClassifier,
        "params": {
            "sel__k": [5, 8, 10, 20, 30, 40, 60]
        }
    },
    "mlp": {
        "classif": MLPClassifier,
        "params": {
            "classif__hidden_layer_sizes": [(20,), (10,)],
            "classif__activation": ['relu'],
            "classif__solver": ['adam'],
            "classif__alpha": [10**i for i in range(-9, -5)],
            "classif__learning_rate": ['constant'],
            "classif__max_iter": [2000],
            "sel__k": ["all"]
        },
    },
    "lda": {
        "classif": LinearDiscriminantAnalysis,
        "params": {
            "sel__k": [5, 8, 10, 20, 30, 40, 60]
        }
    }
}

classif = "lda"
seglen = 10
mode = "dasps"
# domains = ["rel_pow"]
domains = ["rel_pow", "conn", "ai", "time"]
labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
domains += ['abs_pow']

verbosity = 0


def oversample(features, labels, groups):
    label_counts = np.bincount(labels)
    max_label_count = np.max(label_counts)

    for label in np.unique(labels):
        n_to_oversample = max_label_count - label_counts[label]

        if n_to_oversample == 0:
            continue

        # Get indices of samples with given label
        label_indices = np.where(labels == label)[0]

        # Randomly select samples to oversample
        new_samples = np.random.choice(label_indices, n_to_oversample)

        features = np.vstack([features, features[new_samples]])
        labels = np.hstack([labels, labels[new_samples]])
        groups = np.hstack([groups, groups[new_samples]])

    return features, labels, groups


def train_models():
    builder = DatasetBuilder(labeling_scheme)

    df = builder.build_dataset_df(
        seglen, mode=mode, domains=domains,
        p_val_thresh=1)

    print("Working with following features:")
    print(*df.columns)

    orig_label_counts = df['label'].value_counts()

    group_encoder = LabelEncoder()
    groups = group_encoder.fit_transform(df['uniq_subject_id'])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])

    # print(df['label'].value_counts().plot(kind='bar'))
    # plt.show()

    # Remove ID columns
    df.drop(columns=['uniq_subject_id', 'label'], inplace=True)

    features = df.to_numpy()
    features, labels, groups = oversample(features, labels, groups)

    n_feats = features.shape[1]

    print("Number of features:", n_feats)

    # Print mean of each feature
    print("Mean of each feature:")
    print(np.mean(features, axis=0))

    logo = LeaveOneGroupOut()
    kfold = StratifiedKFold(n_splits=10)

    # Select cross validation
    cv = kfold

    # splits = kfold.get_n_splits(features, labels)

    param_grid = GRID[classif]["params"]

    # Make sure that the number of features in the grid is never
    # greater than the number of available features
    _corrected_feat_selection_grid = []

    for i in param_grid["sel__k"]:
        if type(i) == int:
            if i <= n_feats:
                _corrected_feat_selection_grid.append(i)
        else:
            _corrected_feat_selection_grid.append(i)

    param_grid["sel__k"] = _corrected_feat_selection_grid

    # Create pipeline with feature selection and SVM
    pipeline = Pipeline([
        # ('scaler', MinMaxScaler((0, 1))),
        ('scaler', StandardScaler()),
        ('sel', SelectKBest(score_func=f_classif)),
        ('classif', GRID[classif]["classif"]())
    ])

    search = GridSearchCV(pipeline, param_grid,
                          n_jobs=-1, verbose=verbosity, cv=cv)

    search.fit(features, labels)

    best_estimator = search.best_estimator_

    scores = cross_val_score(
        best_estimator, features, labels, groups=groups, cv=cv,
        verbose=verbosity, n_jobs=None)
    # Output
    print("Label counts:")
    print(orig_label_counts)

    print("Label counts after oversampling:")
    print(np.bincount(labels))

    print("Searched parameters grid:")
    param_table = {
        "Parameter": list(param_grid.keys()),
        "Values": [str(values) for values in param_grid.values()]
    }
    print(tabulate(param_table, headers="keys",
          tablefmt="pretty", maxcolwidths=30))

    # Print selected features
    selected_features = search.best_estimator_.named_steps['sel']
    selected_mask = selected_features.get_support()
    print("Number of selected features:", sum(selected_mask))

    # Selected features
    selected_features_names = df.columns[selected_mask]
    selected_features_table = {
        "Index": range(len(selected_features_names)),
        "Feature Name": selected_features_names
    }

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
        "Metric": ["Mean accuracy", "Std accuracy"],
        "Value": [round(i, 2) for i in [scores.mean(), scores.std()]]
    }

    print(tabulate(score_results, headers="keys", tablefmt="pretty"))

# - Use a custom scoring function (for example balanced accuracy)
# - Define a grid of params for SVM
# - Try enabling overlap
# - Remove corellated features


if __name__ == "__main__":
    train_models()
