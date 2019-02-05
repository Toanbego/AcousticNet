"""
Plotting functions for .wav files. They are read through the SoundFile library with
signal amplitude and sample rate.

These plotting functions were originally written by Seth Adams
"""

from matplotlib import pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

def plot_signals(signals, channel='stereo'):
    """
    Plot time signals
    :param signals:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 10))


    fig.suptitle(f'Time Series - {channel}', size=16)

    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft, channel='stereo'):
    """
    Plots the fast-fourier-transform
    :param fft:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 10))

    fig.suptitle(f'Fourier Transforms - {channel}', size=16)

    i = 0
    for x in range(2):
        for y in range(5):

            data = list(fft.values())[i]
            amplitude, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, amplitude)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    """
    Plots the filter banks
    :param fbank:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(mfccs):
    """
    Plots the mel filter cepstrum coefficients
    :param mfccs:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_class_distribution(df, class_dist):
    """
    Plots a cake diagram of the distribution of the various classes
    of the UrbanSound dataset
    :param df:
    :return:
    """
    # Plot the distribution
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, shadow=False, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.show()
    df.reset_index(inplace=True)