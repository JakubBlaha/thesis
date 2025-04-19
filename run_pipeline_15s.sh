#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e
# Treat unset variables as an error
set -u
# Pipeline returns the exit status of the last command that returned a non-zero status
set -o pipefail

echo "----------------------------------------"
echo "Starting 15-second segment ML pipeline"
echo "----------------------------------------"

echo "[1/9] Converting datasets to FIF format"
# python3 -m scripts convert

echo "[2/9] Segmenting the data (15s segments)"
# python3 -m scripts segment --seglen 15

echo "[3/9] Running autoreject"
# python3 -m scripts autoreject

echo "[4/9] Extracting features"
# python3 -m scripts extract --seglen 15

echo "[5/9] Training model (SVM-RBF)"
python3 -m scripts train --labeling-scheme ham --cv logo --seglens 15 --classifiers svm-rbf --no-oversample

echo "[6/9] Calculating metrics for SVM-RBF model"
python3 -m scripts metrics

# echo "[7/9] Running deep learning (LSTM)"
# python3 -m scripts deep --seglen 15 --classif lstm

# echo "[8/9] Training ensemble models (stacking)"
# python3 -m scripts ensemble --strategy stacking --seglen 15

# echo "[9/9] Calculating metrics for ensemble model"
# python3 -m scripts metrics

echo "----------------------------------------"
echo "Pipeline completed successfully!"
echo "----------------------------------------"
