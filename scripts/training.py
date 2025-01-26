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


def train_models():
    global search

    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    df = builder.build_dataset_df(10, mode='dasps')

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
        'svm__kernel': ['linear'],
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

    # Print selected features
    selected_features = search.best_estimator_.named_steps['feature_selection']
    selected_mask = selected_features.get_support()
    print("Number of selected features:", sum(selected_mask))
    print("Selected features:")
    print(df.columns[selected_mask])

    print("Best parameters:", search.best_params_)
    print("Best score:", search.best_score_)

    print("Mean accuracy: ", scores.mean())
    print("Std accuracy: ", scores.std())


# - Use a custom scoring function (for example balanced accuracy)
# - Define a grid of params for SVM
# - Try enabling overlap
# - Remove corellated features


if __name__ == "__main__":
    train_models()
