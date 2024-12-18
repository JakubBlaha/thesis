# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

from utils import DatasetBuilder, LabelingScheme, DaspsLabeling


def train_models():
    global search

    labeling_scheme = LabelingScheme(DaspsLabeling.HAM)
    builder = DatasetBuilder(labeling_scheme)

    df = builder.build_dataset_df(10, mode='dasps')

    # print(df)

    # print(df['label'].value_counts())

    group_encoder = LabelEncoder()
    groups = group_encoder.fit_transform(df['uniq_subject_id'])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])

    # print(df['label'].value_counts().plot(kind='bar'))

    plt.show()

    df.drop(columns=['uniq_subject_id', 'label'], inplace=True)

    # print(df.columns)

    features = df.to_numpy()

    # print(features.shape)
    # print(groups)
    # print(len(set(groups)))

    logo = LeaveOneGroupOut()
    kfold = StratifiedKFold(n_splits=10)

    cv = kfold

    # splits = logo.get_n_splits(features, labels, groups)
    # print(splits)

    param_grid = {
        'C': np.logspace(-2, 10, 13),
        'gamma': np.logspace(-9, 3, 13),
        'kernel': ['rbf'],
    }

    search = GridSearchCV(svm.SVC(), param_grid,
                          n_jobs=None, verbose=10, cv=cv)

    search.fit(features, labels)

    # scores = cross_val_score(clf, features, labels,
    #                          groups=groups, cv=cv, verbose=10, n_jobs=None)

    # print("Mean accuracy: ", scores.mean())
    # print("Std accuracy: ", scores.std())

# - Use a custom scoring function
# - Define a grid of params for SVM


if __name__ == "__main__":
    train_models()
