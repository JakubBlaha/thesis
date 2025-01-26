# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from utils import DatasetBuilder, LabelingScheme, DaspsLabeling
from tabulate import tabulate


def train_models():
    global search

    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    df = builder.build_dataset_df(10, mode='dasps', domains=["time"])

    print("Working with following features:")
    print(*df.columns)

    print("Label counts:")
    print(df['label'].value_counts())

    group_encoder = LabelEncoder()
    groups = group_encoder.fit_transform(df['uniq_subject_id'])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])

    # print(df['label'].value_counts().plot(kind='bar'))
    # plt.show()

    # Remove ID columns
    df.drop(columns=['uniq_subject_id', 'label'], inplace=True)

    features = df.to_numpy()

    # Print mean of each feature
    print("Mean of each feature:")
    print(np.mean(features, axis=0))

    logo = LeaveOneGroupOut()
    kfold = StratifiedKFold(n_splits=10)

    # Select cross validation
    cv = kfold

    splits = kfold.get_n_splits(features, labels)

    print("Splits:", splits)

    param_grid = {
        # 'feature_selection__k': list(range(1, 10, 1)),
        'feature_selection__k': [5, 6, 8, 10, 20, 40, 60],
        # 'feature_selection__k': [8],
        'svm__C': np.logspace(-2, 5, 8),
        'svm__gamma': np.logspace(-9, 0, 10),
        'svm__kernel': ['rbf'],
    }

    # Create pipeline with feature selection and SVM
    pipeline = Pipeline([
        # ('scaler', MinMaxScaler((0, 1))),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('svm', svm.SVC())
    ])

    search = GridSearchCV(pipeline, param_grid,
                          n_jobs=-1, verbose=10, cv=cv)

    search.fit(features, labels)

    best_estimator = search.best_estimator_

    scores = cross_val_score(best_estimator, features, labels,
                             groups=groups, cv=cv, verbose=10, n_jobs=None)

    print("Searched parameters grid:")
    param_table = {
        "Parameter": list(param_grid.keys()),
        "Values": [str(values) for values in param_grid.values()]
    }
    print(tabulate(param_table, headers="keys",
          tablefmt="pretty", maxcolwidths=30))

    # Print selected features
    selected_features = search.best_estimator_.named_steps['feature_selection']
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
        "Metric": ["Best score", "Mean accuracy", "Std accuracy"],
        "Value": [round(i, 3) for i in [search.best_score_, scores.mean(), scores.std()]]
    }

    print(tabulate(score_results, headers="keys", tablefmt="pretty"))

# - Use a custom scoring function (for example balanced accuracy)
# - Define a grid of params for SVM
# - Try enabling overlap
# - Remove corellated features


if __name__ == "__main__":
    train_models()
