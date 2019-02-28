"""
Author - Torstein Gombos
Date - 08.01.2019

Script that is i used to plot and analyze the audio signals.
Various tests can be run in regards to sampling rates, feature extraction and etc.

The script is also used to clean and downsample signals so they are of equal length and size.
"""


import pandas as pd
import argparse

import ffmpeg
from scipy.signal import resample
import resampy

from tqdm import tqdm
import librosa
from scipy.io import wavfile
import soundfile as sf
from python_speech_features import mfcc, logfbank, fbank, sigproc
import wave
import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import axes as ax
import plotting_functions


def parse_arguments():
    """
    Parse arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('--plot', '-p', type=str,
                        help='plot data',
                        default='False')

    parser.add_argument('--clean_files', '-c', type=str,
                        help='activate to downsample and clean files',
                        default='False')

    args = parser.parse_args()
    return args

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


def add_length_to_column(df):
    """
    Goes through all the folds and add the length of each
    signal to the column

    :param df: DataFrame
    :return:
    """
    # Set column to index to loop through it faster
    df.set_index('slice_file_name', inplace=True)

    # Loop through audio files and check the length of the file
    for f, fold in tqdm(zip(df.index, df.fold)):
        signal, rate = sf.read(f'../Datasets/audio/fold{fold}/{f}')
        df.at[f, 'length'] = signal.shape[0] / rate

    df.to_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')
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


def separate_stereo_signal(y):
    """
    Takes a two-channel audio data and separates the audio into
    :param y: The signal
    :return:
    """
    if y.shape[0] > y.shape[1]:
        channels = np.array(y).T

        left_channel = channels[0]
        right_channel = channels[1]

    elif y.shape[1] > y.shape[0]:
        # y = np.reshape(y, (y.shape[1], y.shape[0]))
        channels = np.array(y)

        left_channel = channels[0]
        right_channel = channels[1]

    return left_channel, right_channel


def envelope(y, rate, threshold, dynamic_threshold=True):
    """
    Function that helps remove redundant information from time series
    signals.
    :param y: signal
    :param rate: sampling rate
    :param threshold: magnitude threshold
    :param dynamic_threshold
    :return:
    """

    # Checks one column of the signal dataframe
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 15), min_periods=1, win_type='hamming', center=True).mean()

    mask = np.greater(y_mean, threshold)

    return mask


def make_signal_mono(y):
    """
    If a signal is stereo, average out the channels and make the signal mono
    :return:
    """
    if len(y.shape) > 1:
        y = y.reshape((-1, y.shape[1])).T
        y = np.mean(y, axis=0)
    return y


def resample_signal(y, orig_sr, target_sr):
    """
    Resamples a signal from original samplerate to target samplerate
    :return:
    """

    # 1 - step
    ratio = float(16000) / sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))

    # 2 - step
    y_hat = resampy.resample(y, orig_sr, target_sr, filter='kaiser_best', axis=-1)

    # 3-step
    n = y_hat.shape[-1]

    if n > n_samples:
        slices = [slice(None)] * y_hat.ndim
        slices[-1] = slice(0, n_samples)
        y_hat = y_hat[tuple(slices)]

    elif n < n_samples:
        lengths = [(0, 0)] * y_hat.ndim
        lengths[-1] = (0, n_samples - n)
        y_hat = np.pad(y_hat, lengths, 'constant')

    # 4 - step
    return np.ascontiguousarray(y_hat)


def extract_features(signal, rate, clean=False, fft=False, filterbank=False, mffc=False, dynamic_threshold=False):
    """
    Reads a signal and calculates the fft, filter bank and mfcc.
    Always return the signal.
    :param signal: Audio signal of time
    :param rate: Sample rate of signal
    :param clean: Removes low amplitudes from the signal
    :param fft: Return the frequency and magnitude from the fast fourier transform
    :param filterbank: Return the log filter bank spectrogram
    :param mffc: Return the Mel Frequency Cepstrum
    :param dynamic_threshold:
    :return:
    """

    list_of_returns = list()

    # Clean signal
    if clean is True:
        threshold = 0.006

        # Use dynamic threshold reduction
        if dynamic_threshold is True:
            mask = np.array(envelope(signal, rate, threshold=threshold))

            # If threshold is to high, reduce the threshold
            while mask.sum() < (len(signal)*0.8):
                print("Threshold is to high. Reducing threshold")
                threshold *= 0.95
                mask = np.array(envelope(signal, rate, threshold=threshold))

        else:
            mask = envelope(signal, rate, threshold=threshold, dynamic_threshold=False)
        print(mask.sum())
        # Mask the signal
        signal = signal[mask]

    list_of_returns.append(signal)

    # Calculate fourier transform
    if fft is True:
        fft = calc_fft(signal, rate)
        list_of_returns.append(fft)
    # Find filter bank coefficients
    if filterbank is True:
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        # bank = fbank(signal[:rate], rate, nfilt=26)[0].T
        list_of_returns.append(bank)

    # Find mel frequency
    if mffc is True:
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
        list_of_returns.append(mel)

    return list_of_returns


def plot_data(signals=None, fbank=None, mfccs=None, fft=None):
    """
    Plots the data
    :return:
    """
    # Plot class distribution
    plotting_functions.plot_class_distribution(class_dist)

    # Plot the time signal
    if signals is not None:
        plotting_functions.plot_signals(signals, channel='Stereo')
        plt.show()

    # Plot Fourier Transform
    if fft is not None:
        plotting_functions.plot_fft(fft, channel='Stereo')
        plt.show()

    # Plot the filter banks
    if fbank is not None:
        plotting_functions.plot_fbank(fbank)
        plt.show()

    # Plot the Mel Cepstrum Coefficients
    if mfccs is not None:
        plotting_functions.plot_mfccs(mfccs)
        plt.show()


if __name__ == '__main__':
    """
    Main function
    :return:
    """
    args = parse_arguments()

    # Create dataframe
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')

    # Create a class distribution
    class_dist = df.groupby(['label'])['length'].mean()

    # Fetch the classes from the CSV file
    classes = list(np.unique(df.label))

    # Dicts for things we want to store
    signals = {}
    signals_clean = {}
    signals_left = {}
    signals_right = {}
    mfccs = {}
    filterbank = {}
    fft = {}
    fft_clean = {}
    fft_left = {}
    fft_right = {}

    # Loop through folds and calculate spectrogram and plot data
    i = 0
    c = 'street_music'
    #
    # for c in classes:
    #     # Get the file name and the fold it exists in from the dataframe
    #     wav_file = df[df.label == c].iloc[0, 0]
    #     fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
    #
    #     # Read signal and add it to dict. UrbanSound uses stereo, so there is two channels.
    #     signal, sr = sf.read(f'../Datasets/audio/original/fold{fold}/{wav_file}')
    #     step = int(sr/4)
    #     # Separate the stereo audio
    #     left_channel, right_channel = separate_stereo_signal(signal)
    #
    #     # mask = envelope(right_channel, sr, threshold=0.007)
    #
    #     rand_index = np.random.randint(0, right_channel.shape[0] - step)
    #     right_channel = right_channel[rand_index:rand_index + step]
    #
    #     signals[c], fft[c], filterbank[c], mfccs[c] = extract_features(right_channel, sr,
    #                                                                    clean=False,
    #                                                                    fft=True,
    #                                                                    filterbank=True,
    #                                                                    mffc=True,
    #                                                                    dynamic_threshold=False)
    #
    #     signals_right[c], fft_right[c], filterbank[c], mfccs[c] = extract_features(signal, sr,
    #                                                                                fft=True,
    #                                                                                filterbank=True,
    #                                                                                mffc=True)

    # Plot
    # plt.title(c)
    #
    # plt.plot(signals_right[c], 'b')
    # plt.plot(signals[c], 'r')
    # plt.legend(('Before thresholding', 'After thresholding'))
    # plt.show()
    #
    # plt.title(c)
    # plt.ylim((0, np.max(fft[c][0]*1.1)))
    # plt.xlim(-200, np.max(fft[c][1]))
    # plot1 = plt.plot(fft[c][1], fft[c][0], 'r-',
    #                  # alpha=0.7
    #                  )
    # plot2 = plt.plot(fft_right[c][1], fft_right[c][0], 'b')

    # plt.legend(('After thresholding', 'Before thresholding'))
    # # plt.legend(['Before thresholding'])
    # plt.show()

    # for c in signals:
    #     print(filterbank[c])
    #
    # If true, plot data
    if args.plot == 'True':
        plot_data(signals, filterbank, mfccs, fft=None)

    # Clean and downsample the wav files
    if args.clean_files == 'True':

        # Store in clean directory
        for wav_file in tqdm(df.slice_file_name):

            # Find filename and filepath
            fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
            file_name = f'../Datasets/audio/original/fold{fold}/{wav_file}'

            signal, sr = sf.read(file_name)
            # print(signal.shape[0]/sr)

            signal = make_signal_mono(signal)
            signal_hat = resample_signal(signal, orig_sr=sr, target_sr=16000)
            # print(signal.shape[0] / sr)

            # Filter signal
            mask = envelope(signal_hat, sr, threshold=0.005)
            # plt.ylim(-1, 1)
            #
            # plt.plot(signal_hat[mask])
            # plt.show()
            # exit()

            # Write to file
            wavfile.write(filename=f'../Datasets/audio/new_test/fold{fold}/{wav_file}',
                          rate=16000,
                          data=signal_hat[mask])

            # # FIRST SIGNAL
            # signal, sr = librosa.load(file_name, sr=44100)
            # # signal, _ = separate_stereo_signal(signal)
            # signal1 = librosa.resample(signal, sr, 16000)
            # plt.ylim(-1, 1)
            #
            # plt.plot(signal1)
            #
            # # signal_fft = calc_fft(signal1, sr)
            # # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 10))
            # #
            # # ax1.plot(signal1)
            # # ax2.plot(signal_fft[1], signal_fft[0])
            # plt.show()
            #
            # # Downsample signal
            # # 1 - step
            # # ratio = float(16000) / sr
            # # n_samples = int(np.ceil(signal.shape[-1] * ratio))
            # #
            # # # 2 - step
            # # signal2 = resampy.resample(signal, sr, 16000, filter='kaiser_best', axis=-1)
            # #
            # # # 3-step
            # # n = signal2.shape[-1]
            # #
            # # if n > n_samples:
            # #     slices = [slice(None)] * signal2.ndim
            # #     slices[-1] = slice(0, n_samples)
            # #     signal2 = signal2[tuple(slices)]
            # #
            # # elif n < n_samples:
            # #     lengths = [(0, 0)] * signal2.ndim
            # #     lengths[-1] = (0, n_samples - n)
            # #     signal2 = np.pad(signal2, lengths, 'constant')
            # #
            # # # 4 - step
            # # signal2 = np.ascontiguousarray(signal2)
            # #
            #
            # # signal_fft2 = calc_fft(signal2, 16000)
            # #
            # # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 10))
            # #
            # # ax1.plot(signal2)
            # # ax2.plot(signal_fft2[1], signal_fft2[0])
            #
            # plt.show()
            #
            # # LAST NYTT SIGNAL HER
            # signal, sr = sf.read(file_name)
            #
            # # 1 - step
            # ratio = float(16000) / sr
            # n_samples = int(np.ceil(signal.shape[-1] * ratio))
            #
            # # 2 - step
            # # signal2 = resample(signal, n_samples, axis=-1)
            # signal2 = resampy.resample(signal, sr, 16000, filter='kaiser_best', axis=-1)
            #
            # # 3-step
            # n = signal2.shape[-1]
            #
            # if n > n_samples:
            #     slices = [slice(None)] * signal2.ndim
            #     slices[-1] = slice(0, n_samples)
            #     signal2 = signal2[tuple(slices)]
            #
            # elif n < n_samples:
            #     lengths = [(0, 0)] * signal2.ndim
            #     lengths[-1] = (0, n_samples - n)
            #     signal2 = np.pad(signal2, lengths, 'constant')
            #
            # # 4 - step
            # signal2 = np.ascontiguousarray(signal2)
            #
            # plt.ylim(-1, 1)
            # plt.plot(signal2)
            # plt.show()
            #
            # # print(sr)
            # # if signal.shape[1] > 1:
            # #     signal = signal.reshape((-1, signal.shape[1])).T
            # #     signal = np.mean(signal, axis=0)
            # # # _, signal = separate_stereo_signal(signal)
            # # signal_fft = calc_fft(signal, sr)
            # # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 10))
            # #
            # # ax1.plot(signal, '-r')
            # # ax2.plot(signal_fft[1], signal_fft[0], '-r')
            #
            # # # Downsample signal
            # # signal2 = resample(signal, 16000)
            # # signal_fft2 = calc_fft(signal2, 16000)
            # #
            # # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15, 10))
            # #
            # # ax1.plot(signal2, '-r')
            # # ax2.plot(signal_fft2[1], signal_fft2[0], '-r')
            #
            # # plt.show()
            #
            #
            # # Filter signal
            # mask = envelope(signal, sr, threshold=0.005)
            #
            # # Write to file
            # wavfile.write(filename=f'../Datasets/audio/new_test/fold{fold}/{wav_file}', rate=16000, data=signal)








