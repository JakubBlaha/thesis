import os

TARGET_SAMPLING_FREQ = 128
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7',
                 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# SAD_MULTIPLY_FACTOR = [[0.6965949190168781] * len(CHANNEL_NAMES)]
SAD_MULTIPLY_FACTOR = [
    [0.67687853, 0.826957, 0.68478623, 0.82452861, 0.83293021, 0.53549749,
     0.56795765, 0.56180512, 0.75153026, 0.98081548, 0.58518786, 0.55739797,
     0.72058656, 0.64546991]]

package_dir = os.path.dirname(os.path.abspath(__file__))
features_dir = os.path.join(package_dir, "../data/features")
