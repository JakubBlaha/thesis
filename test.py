# %%
import matplotlib.pyplot as plt
import numpy as np
import mne
from matplotlib import pyplot as plt


def plot(path):
    epochs = mne.read_epochs(path)
    # epochs = epochs.resample(127)
    fig = epochs.plot_psd(fmin=0, fmax=50)
    fig.set_size_inches(20, 12)


plot("data/segmented/10s/clean/S017-epo.fif")
# plot("data/segmented/10s/clean/S103-epo.fif")
# plot("data/segmented/10s/clean/S403-epo.fif")

plt.show()

# %%


def triangle_wave(length, period):
    """
    Generates a triangle wave of specified length and period.

    Args:
        length (int): Length of the triangle wave.
        period (int): Period of the triangle wave.

    Returns:
        numpy.ndarray: Triangle wave.
    """
    x = np.arange(length)
    return 2 * np.abs(2 * (x / period - np.floor(x / period + 0.5))) - 1


# Example usage:
length = 500  # Length of the signal
period = 50  # Period of the triangle wave

triangle_signal = triangle_wave(length, period)

# Plotting the generated triangle wave
plt.figure(figsize=(10, 6))
plt.plot(triangle_signal)
plt.title("Triangle Wave Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# FFT and plot
fft = np.fft.fft(triangle_signal)
frequencies = np.fft.fftfreq(len(fft))

plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fft))
plt.title("FFT of Triangle Wave Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

# %%
arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 10
fft = np.fft.fft(arr)

plt.figure(figsize=(10, 6))
plt.plot(np.abs(fft))
plt.title("FFT of Impulses")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


# %%
path = "S01.edf"

raw = mne.io.read_raw_edf(path, preload=True)
# raw.plot_psd(fmin=0, fmax=64, dB=False)

channels_info = raw.info['ch_names']
print(channels_info)

channels = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4',
    'F8', 'AF4', 'RAW_CQ', 'CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7',
    'CQ_P7', 'CQ_O1', 'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8',
    'CQ_AF4']
raw = raw.pick_channels(channels)

raw = raw.filter(l_freq=0.5, h_freq=50)
raw = raw.set_eeg_reference("average", projection=False)
raw = raw.drop_channels(
    ["RAW_CQ", 'CQ_AF3', 'CQ_F7', 'CQ_F3', 'CQ_FC5', 'CQ_T7', 'CQ_P7', 'CQ_O1',
     'CQ_O2', 'CQ_P8', 'CQ_T8', 'CQ_FC6', 'CQ_F4', 'CQ_F8', 'CQ_AF4'])
raw.set_montage("standard_1020")
raw.plot_psd(fmin=0, fmax=64)

# %%
a = raw.pick_channels(['AF3'])
# a = a.filter(l_freq=0.5, h_freq=60)
a.plot(scalings=1e-4, block=True)
