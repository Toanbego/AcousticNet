"""
Author - Torstein Gombos
Date - 08.01.2019

This script reads CSV files and uses JSON to to navigate the data that exists on YouTube.
"""

import fnmatch
import json
import pandas as pd
import wave
from glob import glob
from tqdm import tqdm
import os
import librosa
import soundfile as sf
from scipy import fftpack
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import random
import numpy as np
import csv
import re
from matplotlib import pyplot as plt
import plotting_functions




def fetch_labels_youtube(file_path="../Datasets/balanced_train_segments.csv"):
    """
    Load the csv file for the YouTube data set
    :return:
    """

    with open(file_path, 'r') as csvfile:
        df = csv.reader(csvfile, delimiter=' ', quotechar='"')
        i = 0
        for row in df:
            i += 1
            print(', '.join(row))
        print(f'There is {i} samples in the data')
        print(df)


def find_audio_files_path():
    """
    Creates the path to all the wav files in the various
    folds.
    :return:
    """
    files = []
    start_dir = '../Datasets/audio/'
    pattern = "*.wav"

    for dir, _, _ in os.walk(start_dir):
        files.extend(glob(os.path.join(dir, pattern)))
    return files


def add_length_to_column(df):
    """
    Goes through all the folds and add the length of each
    signal to the column
    :return:
    """

    # for f in tqdm(files):
    #     filename = re.search('\\\\(.*)', f)[0][1:]
    #     s, rate = sf.read(f)
    #     signal_length = s.shape[0] / rate
    #     df.at[filename, 'length'] = signal_length

    for f, fold in tqdm(zip(df.index, df.fold)):
        signal, rate = sf.read(f'../Datasets/audio/fold{fold}/{f}')
        df.at[f, 'length'] = signal.shape[0] / rate
    return df


def calc_fft(y, rate):
    """
    Calculates the fast fourier transform given signal and sampling rate
    :param y: The signal
    :param rate: Sampling rate
    :return:
    """
    n = len(y)
    freq = librosa.fft_frequencies(sr=rate, n_fft=n)

    magnitude = abs(np.fft.rfft(y) / n)

    return magnitude, freq


def separate_stereo_signal(signal):
    """
    Takes a two-channel audio data and separates the audio into
    :param signal:
    :return:
    """
    left_channel = []
    right_channel = []
    for sample in signal:
        left_channel.append(sample[0])
        right_channel.append(sample[1])

    return left_channel, right_channel


def envelope(y, rate, threshold):
    """
    Function that helps remove redundant information from time series
    signals.
    :param y: signal
    :param rate: sampling rate
    :param threshold: magnitude threshold
    :return:
    """
    mask = []
    y = pd.Series(y[y.columns[0]]).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


if __name__ == '__main__':
    """
    Main function
    :return:
    """
    # Create dataframe
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K.csv')


    # Format data and add a 'length' column
    df.set_index('slice_file_name', inplace=True)
    df = add_length_to_column(df)

    # Create a class distribution
    class_dist = df.groupby(['label'])['length'].mean()
    plotting_functions.plot_class_distribution(df, class_dist)

    classes = list(np.unique(df.label))

    # Dicts
    signals = {}
    signals_clean = {}
    signals_left = {}
    signals_right = {}
    mfccs = {}
    fbank = {}
    fft = {}
    fft_clean = {}
    fft_left = {}
    fft_right = {}

    # Loop through folds and calculate spectrogram and plot data
    for c in classes:
        # Get the file name and the fold it exists in from the dataframe
        wav_file = df[df.label == c].iloc[0, 0]
        fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]

        # Read signal and add it to dict. UrbanSound uses stereo, so there is two channels.
        signal, sr = sf.read(f'../Datasets/audio/fold{fold}/{wav_file}')

        # Separate the stereo audio
        left_channel, right_channel = separate_stereo_signal(signal)
        df_signal = pd.DataFrame({'left_channel': left_channel, 'right_channel': right_channel})

        # Create envelope for signal
        mask = envelope(df_signal, sr, threshold=0.009)
        signal_clean = np.array(left_channel)[mask]

        signals[c] = signal
        signals_left[c] = left_channel
        signals_right[c] = right_channel
        signals_clean[c] = signal_clean

        # Find the fast-fourier-transform
        fft[c] = calc_fft(signal, sr)
        fft_clean[c] = calc_fft(left_channel, sr)
        fft_left[c] = calc_fft(signal_clean, sr)
        fft_right[c] = calc_fft(right_channel, sr)

        # Find filter bank coefficients
        bank = logfbank(signal[:sr], sr, nfilt=26, nfft=1103).T
        fbank[c] = bank

        # Find mel frequency
        mel = mfcc(signal[:sr], sr, numcep=13, nfilt=26, nfft=1103).T
        mfccs[c] = mel

    # Plot the time signal
    plotting_functions.plot_signals(signals_left, channel='left_channel')
    plt.show()
    # plotting_functions.plot_signals(signals_right, channel='right_channel')
    # plt.show()
    plotting_functions.plot_signals(signals_clean, channel='cleaned_left_channel')
    plt.show()
    # plotting_functions.plot_signals(signals, channel='stereo')
    # plt.show()

    # Plot Fourier Transform
    plotting_functions.plot_fft(fft_left, channel='left_channel')
    plt.show()
    plotting_functions.plot_fft(fft_clean, channel='left_channel CLEAN')
    plt.show()
    # plotting_functions.plot_fft(fft_right, channel='right_channel')
    # plt.show()

    # # Plot the filter banks
    # plotting_functions.plot_fbank(fbank)
    # plt.show()
    #
    # # Plot the Mel Cepstrum Coefficients
    # plotting_functions.plot_mfccs(mfccs)
    # plt.show()






