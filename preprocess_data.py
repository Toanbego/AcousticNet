"""
Author - Torstein Gombos
Date - 08.01.2019

This script reads CSV files and uses JSON to to navigate the data that exists on YouTube.
"""


import pandas as pd
import argparse
import ffmpeg

from tqdm import tqdm
import os
import librosa
from scipy.io import wavfile
import wave
import soundfile as sf
from python_speech_features import mfcc, logfbank
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
                        help='Either conv or time',
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
    channels = np.array(y).T
    left_channel = channels[0]
    right_channel = channels[1]

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

    # Checks one column of the signal dataframe
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()

    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def extract_features(signal, rate, clean=False ,fft=False, fbank=False, mffc=False, config=None):
    """
    Reads a signal and calculates the fft, filter bank and mfcc.
    Always return the signal.
    :param signal: Audio signal of time
    :param rate: Sample rate of signal
    :param clean: Removes low amplitudes from the signal
    :param fft: Return the frequency and magnitude from the fast fourier transform
    :param fbank: Return the log filter bank spectrogram
    :param mffc: Return the Mel Frequency Cepstrum
    :return:
    """

    list_of_returns = list()

    list_of_returns.append(signal)

    # Clean signal
    if clean is True:
        mask = envelope(signal, rate, threshold=0.0005)
        list_of_returns.append(signal[mask])

    # Calculate fourier transform
    if fft is True:
        fft = calc_fft(signal, rate)
        list_of_returns.append(fft)
    # Find filter bank coefficients
    if fbank is True:
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
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
    fbank = {}
    fft = {}
    fft_clean = {}
    fft_left = {}
    fft_right = {}

    # Loop through folds and calculate spectrogram and plot data
    c = 'children_playing'
    # for c in classes:
    # Get the file name and the fold it exists in from the dataframe
    wav_file = df[df.label == c].iloc[0, 0]
    fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]

    # Read signal and add it to dict. UrbanSound uses stereo, so there is two channels.
    signal, sr = sf.read(f'../Datasets/audio/downsampled/fold{fold}/{wav_file}')

    # Separate the stereo audio
    left_channel, right_channel = separate_stereo_signal(signal)

    mask = envelope(right_channel, sr, threshold=0.005)

    signals[c], fft[c], fbank[c], mfccs[c] = extract_features(right_channel[mask], sr,
                                                              fft=True,
                                                              fbank=True,
                                                              mffc=True)

    signals_right[c], fft_right[c], fbank[c], mfccs[c] = extract_features(right_channel, sr,
                                                              fft=True,
                                                              fbank=True,
                                                              mffc=True)

    # Plot
    plt.title('Children playing')

    plt.plot(signals_right[c], 'b')
    plt.plot(signals[c], 'r')
    plt.legend(('Before thresholding', 'After thresholding'))
    plt.show()

    plt.title('Children playing')
    plt.ylim((0, 0.00150))
    plt.xlim(-200, 2000)
    plot1 = plt.plot(fft[c][1], fft[c][0], 'r-',
                     # alpha=0.7
                     )
    plot2 = plt.plot(fft_right[c][1], fft_right[c][0], 'b')

    plt.legend(('After thresholding', 'Before thresholding'))
    # plt.legend(['Before thresholding'])
    plt.show()



    # If true, plot data
    # if args.plot == 'True':
    #     plot_data(signals, fbank, mfccs, fft)

    # Store in clean directory
    # for wav_file in tqdm(df.slice_file_name):
    #     # Reads a down sampled signal
    #     fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
    #
    #     # Check if there are files in clean folder from before
    #     if len(os.listdir('../Datasets/audio/fold{fold}')) != 0:
    #         continue
    #     else:
    #         signal, sr = sf.read(f'../Datasets/audio/fold{fold}/{wav_file}')
    #
    #         # sr, signal = wave.wave(f'../Datasets/audio/fold{fold}/{wav_file}', sr=16000)
    #
    #         wavfile.write(filename=f'../Datasets/audio/fold{fold}/{wav_file}', rate=16000, data=signal)
    #         # mask = envelope(signal, rate, threshold=0.0005)
    #         # sf.write(file=f'../Datasets/audio/clean/fold{fold}/{wav_file}', data=signal, samplerate=sr)







