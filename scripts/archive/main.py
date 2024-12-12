from enum import Enum
import numpy as np
import contextlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn.utils
from torch.utils.data import Dataset
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from imblearn.over_sampling import RandomOverSampler

from common import Trial, TrialLabel
from scripts.dasps_preprocess import DaspsPreprocessor


class FeatureGroup(Enum):
    TIME = 0
    REL_POWER = 1
    ABS_POWER = 2
    CONNECTIVITY = 3
    ASYMMETRY = 4


ALL_FEAT_GROUPS = [FeatureGroup.TIME, FeatureGroup.REL_POWER,
                   FeatureGroup.ABS_POWER, FeatureGroup.CONNECTIVITY, FeatureGroup.ASYMMETRY]


class CustomDataset(Dataset):
    def __init__(self, trials: list[Trial]) -> None:
        self.trials = trials

        self._feat_selector = None

        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None

        self._subset_mode = None

    def _get_labels(self):
        return [trial.trial_label.value for trial in self.trials]

    def compute_features(self, groups=ALL_FEAT_GROUPS):
        for index, trial in enumerate(self.trials):
            trial.features = {}

            print("Computing features for trial: ", index)

            if FeatureGroup.TIME in groups:
                trial.compute_time_measures()
            if FeatureGroup.REL_POWER in groups:
                trial.compute_powers(normalize=True)
            if FeatureGroup.ABS_POWER in groups or FeatureGroup.ASYMMETRY in groups:
                trial.compute_powers(normalize=False)
            if FeatureGroup.CONNECTIVITY in groups:
                trial.compute_connectivity()
            if FeatureGroup.ASYMMETRY in groups:
                trial.compute_asymmetry()

            if FeatureGroup.ASYMMETRY in groups and FeatureGroup.ABS_POWER not in groups:
                trial.features = {key: val for key, val in trial.features.items() if 'abs_pow' not in key}

        self._train_data = None
        self._train_labels = None
        self._train_data = None
        self._train_labels = None

    def select_by_var(self):
        self._feat_selector = VarianceThreshold(threshold=0.01)

    def select_k_best(self, k=11):
        self._feat_selector = SelectKBest(k=k)

    def select_all_features(self):
        self._feat_selector = None

    def preprocess(self, oversample=True, normalize=True, shuffle=True):
        data = np.array([list(trial.features.values()) for trial in self.trials])
        labels = [trial.trial_label.value for trial in self.trials]

        # Select features
        _feat_names = list(self.trials[0].features.keys())

        _transform_args: list = [data]

        if self._feat_selector:
            if isinstance(self._feat_selector, SelectKBest):
                _transform_args.append(labels)

            self._feat_selector.fit_transform(*_transform_args)

            _sel_feat_names = self._feat_selector.get_feature_names_out(_feat_names)
            _sel_feat_idx = self._feat_selector.get_support(indices=True)
        else:
            _sel_feat_names = _feat_names
            _sel_feat_idx = [i for i in range(len(_feat_names))]

        data = data[:, _sel_feat_idx]

        # Normalization
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        # Oversampling
        if oversample:
            ros = RandomOverSampler(random_state=0)
            data, labels = ros.fit_resample(data, labels)

        if shuffle:
            data, labels = sklearn.utils.shuffle(data, labels, random_state=0)

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels)

        self._train_data, self._test_data, self._train_labels, self._test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=0)

    def _get_data_labels_in_context(self):
        if self._subset_mode == 'train':
            return self._train_data, self._train_labels

        if self._subset_mode == 'test':
            return self._test_data, self._test_labels

        raise Exception('No test or train mode selected!')

    @property
    def n_features(self):
        if self._train_data is None:
            raise Exception("Preprocess data first!")

        return self._train_data.shape[1]

    @property
    def data(self):
        _data, _ = self._get_data_labels_in_context()

        if _data is None:
            raise Exception("Preprocess data first!")

        return _data

    @property
    def labels(self):
        _, _labels = self._get_data_labels_in_context()

        if _labels is None:
            raise Exception("Preprocess data first!")

        return _labels

    @property
    def all_data(self):
        return np.concatenate((self._train_data, self._test_data))

    @property
    def all_labels(self):
        return np.concatenate((self._train_labels, self._test_labels))

    @contextlib.contextmanager
    def train(self):
        try:
            self._subset_mode = 'train'
            yield None
        finally:
            self._subset_mode = None

    @contextlib.contextmanager
    def test(self):
        try:
            self._subset_mode = 'test'
            yield None
        finally:
            self._subset_mode = None

    def __len__(self):
        if self._subset_mode is None:
            raise Exception("Train or test mode needs to be selected!")

        _data, _ = self._get_data_labels_in_context()

        if _data is None:
            raise Exception("Preprocess data first!")

        return len(_data)

    def __getitem__(self, idx):
        _data, _labels = self._get_data_labels_in_context()

        if _data is None or _labels is None:
            raise Exception("Preprocess data first!")

        return _data[idx], _labels[idx]


if __name__ == '__main__':
    anx_epochs = DaspsPreprocessor.get_severe_moderate_epochs()
    n_epochs = anx_epochs.get_data().shape[0]

    # conn = np.empty((n_epochs, n_channels, n_channels, n_bands))

    for i in range(n_epochs):
        print(i)

        trial = Trial(TrialLabel.CONTROL, anx_epochs[i])
        trial.compute_all_features()
        # trial.compute_time_measures()

        # conn[i] = trial.compute_connectivity()
        # trial.compute_rel_powers()

        # print(*[(key, trial.features[key])
        #       for key in trial.features], sep='\n')

        print("Total features: ", len(trial.features))

        trial.select_features()

        break

# spectral entropy, sample entropy, approximate entropy, lyapunov exponent, hurst exponent, higuchi fd

# All features:
# Freq: PSD, absolute power, relative power, mean power, asymmetry, band power ratios, HHT
# Time: hjorth, katz fd, higuchi fd, entropy, (lyapunov exponent), (detrended fluctuation analysis), hurst exp
# Time-frequency: DWT, RMS, energy
# Connectivity: PLI, wPLI, graph theory measures


# Classifiers: SVM, RF, Bagging, LightGBM, XGBoost, CatBoost, KNN, LDA, AdaBoost, Gradient Bagging, DT, MLP, SSAE, CNN, DBN, RBFNN, CNN, LSTM, NB
