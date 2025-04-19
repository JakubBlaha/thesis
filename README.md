# Documentation for classification pipeline for thesis on the topic of _EEG-BASED CLASSIFICATION OF ANXIETY SUBTYPES USING MACHINE LEARNING_

Author: Jakub Bláha, xblaha36@stud.fit.vutbr.cz

## Installation

1. Install Python 3.12.7
2. Install Pipenv:
   ```
   python3 -m pip install pipenv
   ```
3. Navigate to the project directory, e.g.:
   ```
   cd ~/thesis
   ```
4. Create a virtual environment and install dependencies:
   ```
   pipenv install
   ```
5. Run pipenv shell before using the CLI:
   ```
   pipenv shell
   ```

## Dataset preparation

The DASPS dataset individual preprocessed `.mat` files need to be placed in `data/datasets/DASPS`. The files can be found in the following directory after downloading and extracting the DASPS archive from https://ieee-dataport.org/open-access/dasps-database:

- `DASPS_Database/Preprocessed data.mat`

The SAD dataset individual files are intended to be placed in `data/datasets/SAD/preprocessed/control` and `data/datasets/SAD/preprocessed/severe` directories. Only these severities are used currently. The files can be found in the following directories after downloading and extracting the provided archive (https://vutbr.sharepoint.com/:f:/s/DataSets/EncrNE2S6etLhttYQEIevyoBipy3Thy_s_ZA2BwfujHmsA?e=M3loxP):

- `Subjects_Hakim_SAD/Control/Eyes close/preprocessed`
- `Subjects_Hakim_SAD/Severe/Eyes close/preprocessed`

The folder data structures should look like the following:

```
data
├── DASPS.csv
└── datasets
    ├── DASPS
    │   ├── S01preprocessed.mat
    │   ├── ...
    │   └── S23preprocessed.mat
    └── SAD
        └── preprocessed
            ├── control
            │   ├── C1.edf
            │   ├── ...
            │   └── C9.edf
            └── severe
                ├── C1.edf
                ├── ...
                └── C9.edf
```

## Command Line Interface (CLI)

The project provides a command-line interface with several commands for the EEG classification pipeline:

### Available Commands

1. **convert** - Convert datasets to FIF format
   ```
   python -m scripts convert
   ```

2. **segment** - Segment data into specified lengths
   ```
   python -m scripts segment --seglen <SEGMENT_LENGTH>
   ```
   - `--seglen`: Required. Segment length in seconds (valid values: 1, 2, 3, 5, 10, 15, 30)

3. **autoreject** - Run autoreject artifact rejection
   ```
   python -m scripts autoreject
   ```

4. **extract** - Extract features from segments
   ```
   python -m scripts extract --seglen <SEGMENT_LENGTH>
   ```
   - `--seglen`: Required. Segment length in seconds

5. **train** - Train machine learning models
   ```
   python -m scripts train --labeling-scheme <ham|sam> --cv <logo|skf> --seglens <LENGTHS> --classifiers <CLASSIFIERS> [options]
   ```
   - `--labeling-scheme`: Required. DASPS labeling scheme (ham or sam)
   - `--cv`: Required. Cross-validation strategy (logo or skf)
   - `--seglens`: Required. Comma-separated list of segment lengths (e.g., '1,2,5')
   - `--classifiers`: Required. Comma-separated list of classifiers (e.g., 'svm-rbf,rf,knn')
   - `--domains`: Optional. Comma-separated list of feature domains (default: rel_pow,conn,ai,time,abs_pow)
   - `--mode`: Optional. Dataset mode (both, dasps, or sad, default: both)
   - `--no-oversample`: Optional. Disable oversampling (enabled by default)

6. **deep** - Run deep learning models
   ```
   python -m scripts deep --seglen <SEGMENT_LENGTH> --classif <lstm|cnn>
   ```
   - `--seglen`: Required. Segment length in seconds
   - `--classif`: Required. Deep learning classifier type (lstm or cnn)

7. **ensemble** - Train ensemble models
   ```
   python -m scripts ensemble --strategy <voting|stacking> [options]
   ```
   - `--strategy`: Required. Ensemble strategy (voting or stacking)
   - `--seglen`: Optional. Segment length in seconds (default: 15)
   - `--mode`: Optional. Dataset mode (both, dasps, or sad, default: both)
   - `--domains`: Optional. Comma-separated list of domains (default: all domains)
   - `--no-oversample`: Optional. Disable oversampling (enabled by default)
   - `--final-classifier`: Optional. Final classifier for stacking (logistic, rf, mlp, or gb, default: logistic)
   - `--seed`: Optional. Random seed (default: 42)

### Complete Pipeline Example

To run the complete pipeline with 15-second segments:

```bash
# Convert datasets to FIF format
python3 -m scripts convert

# Segment the data
python3 -m scripts segment --seglen 15

# Run autoreject
python3 -m scripts autoreject

# Extract features
python3 -m scripts extract --seglen 15

# Train models
python3 -m scripts train --labeling-scheme sam --cv logo --seglens 15 --classifiers svm-rbf,rf,knn

# Run deep learning
python3 -m scripts deep --seglen 15 --classif lstm

# Train ensemble models
python3 -m scripts ensemble --strategy stacking --seglen 15
```

## Citation

> BLÁHA, Jakub. EEG-based classification of anxiety subtypes using machine learning. Online, bachelor's Thesis. Muhammad Asad ZAHEER (supervisor). Brno: Brno University of Technology, Faculty of Information Technology, 2025. Available at: https://www.vut.cz/en/students/final-thesis/detail/161455. [accessed 2025-04-19].