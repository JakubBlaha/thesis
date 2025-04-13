# Author: Jakub Bláha, xblaha36

"""
Constants module for EEG data processing across multiple anxiety datasets.

This module contains constants needed for processing EEG data from DASPS and SAD datasets,
including sampling frequency, channel names, and amplitude normalization factors.

Constants:
    TARGET_SAMPLING_FREQ (int): Target sampling frequency (128 Hz) for resampling all EEG data.
    CHANNEL_NAMES (list): 14 EEG channel names common to both datasets.
    SAD_MULTIPLY_FACTOR (list): Electrode power ratios used to normalize signal amplitudes 
        between datasets. These factors represent the ratio of mean absolute amplitudes
        from low-anxiety subjects in DASPS vs. SAD datasets:
        R_e = mean(|X^DASPS_e|) / mean(|X^SAD_e|)
        
        These ratios are used to scale SAD dataset amplitudes to match DASPS levels:
        X^SAD_e(corrected) = X^SAD_e × R_e
        
        This normalization ensures that equipment-related amplitude differences don't
        introduce artificial distinctions between anxiety groups during classification.
    
    features_dir (str): Path to the directory containing extracted features.

Notes:
    The amplitude normalization process was implemented to reduce significant differences
    found when comparing low-anxiety individuals across datasets. By equalizing signal
    properties of low-anxious individuals from both datasets, true anxiety-related
    differences are better preserved.
"""

import os

TARGET_SAMPLING_FREQ = 128

# Channel names common for both datasets
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7',
                 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# SAD dataset amplitudes should be multiplied by these factors
SAD_MULTIPLY_FACTOR = [
    [0.67687853, 0.826957, 0.68478623, 0.82452861, 0.83293021, 0.53549749,
     0.56795765, 0.56180512, 0.75153026, 0.98081548, 0.58518786, 0.55739797,
     0.72058656, 0.64546991]]

package_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.join(package_dir, "../data/features")
