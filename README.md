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

A comprehensive help can be displayed using the following commands:

```bash
# Display available commands
python3 -m scripts --help

# Display help for a specific command
python3 -m scripts train -h
```

### Available Commands

**`convert`** - Convert datasets to FIF format
   ```
   python -m scripts convert
   ```

**`segment`** - Segment data into specified lengths
   ```
   python -m scripts segment --seglen <SEGMENT_LENGTH>
   ```
   
   | Parameter  | Description                                                      |
   | ---------- | ---------------------------------------------------------------- |
   | `--seglen` | Segment length in seconds (valid values: 1, 2, 3, 5, 10, 15, 30) |

**`autoreject`** - Run autoreject artifact rejection
   ```
   python -m scripts autoreject
   ```

**`extract`** - Extract features from segments
   ```
   python -m scripts extract --seglen <SEGMENT_LENGTH>
   ```
   
   | Parameter  | Description               |
   | ---------- | ------------------------- |
   | `--seglen` | Segment length in seconds |

**`train`** - Train machine learning models
   ```
   python -m scripts train --labeling-scheme <ham|sam> --cv <logo|skf> --seglens <LENGTHS> --classifiers <CLASSIFIERS> [options]
   ```
   
   | Parameter           | Description                                                                     |
   | ------------------- | ------------------------------------------------------------------------------- |
   | `--labeling-scheme` | DASPS labeling scheme (ham or sam)                                              |
   | `--cv`              | Cross-validation strategy (logo or skf)                                         |
   | `--seglens`         | Comma-separated list of segment lengths (e.g., '1,2,5')                         |
   | `--classifiers`     | Comma-separated list of classifiers (e.g., 'svm-rbf,rf,knn')                    |
   | `--domains`         | Comma-separated list of feature domains (default: rel_pow,conn,ai,time,abs_pow) |
   | `--mode`            | Dataset mode (both, dasps, or sad, default: both)                               |
   | `--no-oversample`   | Disable oversampling (enabled by default)                                       |

**`deep`** - Run deep learning models
   ```
   python -m scripts deep --seglen <SEGMENT_LENGTH> --classif <lstm|cnn>
   ```
   
   | Parameter   | Description                                 |
   | ----------- | ------------------------------------------- |
   | `--seglen`  | Segment length in seconds                   |
   | `--classif` | Deep learning classifier type (lstm or cnn) |

**`ensemble`** - Train ensemble models
   ```
   python -m scripts ensemble --strategy <voting|stacking> [options]
   ```
   
   | Parameter            | Description                                                                 |
   | -------------------- | --------------------------------------------------------------------------- |
   | `--strategy`         | Ensemble strategy (voting or stacking)                                      |
   | `--seglen`           | Segment length in seconds (default: 15)                                     |
   | `--mode`             | Dataset mode (both, dasps, or sad, default: both)                           |
   | `--domains`          | Comma-separated list of domains (default: all domains)                      |
   | `--no-oversample`    | Disable oversampling (enabled by default)                                   |
   | `--final-classifier` | Final classifier for stacking (logistic, rf, mlp, or gb, default: logistic) |
   | `--seed`             | Random seed (default: 42)                                                   |

**`metrics`** - Calculate and visualize metrics for classification results
   ```
   python -m scripts metrics [--file <RESULTS_FILE>] [--title <PLOT_TITLE>]
   ```
   
   | Parameter | Description                                                     |
   | --------- | --------------------------------------------------------------- |
   | `--file`  | Path to the results CSV file (uses latest file if not provided) |
   | `--title` | Title for the confusion matrix plot                             |

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
python3 -m scripts train --labeling-scheme ham --cv logo --seglens 15 --classifiers svm-rbf

# Display performance metrics 
python3 -m scripts metrics

# Run deep learning
python3 -m scripts deep --seglen 15 --classif lstm

# Train ensemble models
python3 -m scripts ensemble --strategy stacking --seglen 15

# Display performance metrics 
python3 -m scripts metrics
```

## Citation

> BLÁHA, Jakub. EEG-based classification of anxiety subtypes using machine learning. Online, bachelor's Thesis. Muhammad Asad ZAHEER (supervisor). Brno: Brno University of Technology, Faculty of Information Technology, 2025. Available at: https://www.vut.cz/en/students/final-thesis/detail/161455.