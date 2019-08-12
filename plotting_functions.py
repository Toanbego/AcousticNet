"""
Plotting functions for .wav files. They are read through the SoundFile library with
signal amplitude and sample rate.

These plotting functions were originally written by Seth Adams
"""

from matplotlib import pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import numpy as np
import pandas as pd
import seaborn as sns # data visualization library
import re
# plt.style.use('ggplot')
#

def hamming_window():
    """
    Plot a standard hamming window. It is used to move along a signal after it
    has been sliced into frames.
    :return:
    """
    # Hamming function
    frames = np.zeros(200)
    x = np.linspace(0, 200, 200)
    print(np.shape(frames), np.shape(x))

    hamming_function = []

    for n in x:
        hamming_function.append(0.54 - 0.46 * np.cos((2 * np.pi * n) / (200 - 1)))
    plt.title("Hamming window")
    plt.plot(x, np.array(hamming_function))

    plt.savefig("../Referansebilder/Hamming_window.png")
    plt.show()
    plt.close()


def smoothTriangle(data, degree, dropVals=False):
    triangle = np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))
    if dropVals:
        return smoothed
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def plot_performance(file_path, smoothing, plot_title='Performance', column_name='Value', color=None, style=None, f='f'):
    """
    Plots the csv files from the tensorboard after training a network
    :param column_name:
    :param plot_title:
    :param file_path:
    :param smoothing: smothing factor (integer)
    :return:
    """
    epoch_length = pd.read_csv(f)
    epoch_length = len(epoch_length['Value'])

    # Read csv
    if type(file_path) is str:
        tensorboard_result = pd.read_csv(file_path)
    else:
        tensorboard_result = file_path
    # Apply smoothing if smoothing is given
    if smoothing >= 1:
        tensorboard_result = tensorboard_result.rolling(window=smoothing)
        tensorboard_result = tensorboard_result.mean()

    # Change column name from value to specified feature
    tensorboard_result.rename(inplace=True, columns={'Value': column_name})

    # Convert steps axis to number of epochs
    step_axis = np.linspace(0, 30, len(tensorboard_result['Step']))

    # Plot results
    if color is not None and style is None:
        plt.plot(step_axis, tensorboard_result[column_name], label=column_name, color=color)
    elif style is not None and color is None:
        plt.plot(step_axis, tensorboard_result[column_name], label=column_name, linestyle=style)
    elif style is not None and color is not None:
        plt.plot(step_axis, tensorboard_result[column_name], label=column_name, linestyle=style, color=color)
    else:
        plt.plot(step_axis, tensorboard_result[column_name], label=column_name)


def apply_smoothing(data, smoothing):
    # Apply smoothing if smoothing is given
    if smoothing >= 1:
        data = data.rolling(window=smoothing)
        data = data.mean()
    return data

def plot_results():
    """
    Hardcoded function that plots different metric with correct axis names and numbers
    :return:
    """
    mode = 'box'

    f =   r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\engine_idling\spectrogram\run-.-tag-' + f'val_acc.csv'
    for metric in ['loss', 'val_acc', 'val_loss']:

        # ' + f'{metric}.csv'
        file_path_clean =           r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\augmentation\librosa2\clean\run-.-tag-' + f'{metric}.csv'
        file_path_time_shift_add =      r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\augmentation\librosa_add\time shift\run-.-tag-' + f'{metric}.csv'
        file_path_pitch_shift_add =   r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\augmentation\librosa_add\pitch shift\run-.-tag-' + f'{metric}.csv'
        file_path_noise05_add =   r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\augmentation\librosa_add\noise 0.005\run-.-tag-' + f'{metric}.csv'
        file_path_nosie01 =      r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\augmentation\librosa_add\noise 0.001\run-.-tag-' + f'{metric}.csv'
        file_path_asym_msfb =       r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\asym_reg\scalogram delta\run-.-tag-' + f'{metric}.csv'
        file_path_clean_scalogram = r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\asym_reg\scalogram clean\run-.-tag-' + f'{metric}.csv'
        file_path_asym_scalogram =       r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\asym_reg\scalogram asym\run-.-tag-' + f'{metric}.csv'

        smoothing = 1
        csv_files = [file_path_clean,
                     file_path_time_shift_add,
                     file_path_pitch_shift_add,
                     file_path_noise05_add,
                     file_path_nosie01,
                     # file_path_asym_msfb,
                     # file_path_clean_scalogram,
                     # file_path_asym_scalogram
                     ]
        bikkje = ['replace', 'add']
        # values_only = pd.merge(mfcc, scalogram, on='Step')
        values_only = {}
        if mode == 'box':
            for i, file in enumerate(csv_files):
                if i == 0:
                    feature = re.findall(r"librosa2\\(.*)\\run", file)[0]
                else:
                    feature = re.findall(r"librosa_add\\(.*)\\run", file)[0]
                data = apply_smoothing(pd.read_csv(file).Value, smoothing)
                if feature == 'librosa':
                    feature = 'msfb - 128'
                elif feature == 'librosa60':
                    feature = 'msfb - 128'
                elif feature == 'msfb2':
                    feature = 'msfb - 26'
                elif feature == 'scalogram2':
                    feature = 'scalogram'
                elif feature == 'spectogram' or feature == 'spectogram2':
                    feature = 'spectrogram'
                elif feature == 'mfcc2':
                    feature = 'mfcc'
                elif feature == 'noise_0005' or feature == 'noise 0.005 (add)':
                    feature = '\u03BC = 0.005'

                elif feature == 'noise':
                    feature = '\u03BC = 0.005'
                elif feature == 'noise_0001' or feature == 'noise 0.001 (add)':
                    feature = '\u03BC = 0.001'
                elif feature == 'pitch_shift2':
                    feature = 'pitch shift'
                if feature == 'noise 0.005':
                    data = data
                values_only[feature] = data
            values_only = pd.DataFrame(values_only)
        else:
            smoothing = 12
            # bikkje = ['replace', 'add']
            for i, file in enumerate(csv_files):
                feature = re.findall(r"librosa_add\\(.*)\\run", file)[0]
                # feature = bikkje[i]
                if feature == 'clean':
                    plot_performance(file,
                                     smoothing=smoothing,
                                     column_name=feature,
                                     style='-',
                                     f=f)
                else:
                    plot_performance(file,
                                     smoothing=smoothing,
                                     column_name=feature,
                                     style='--',
                                     f=f)

        if mode == 'box':
            if metric == 'val_acc' or metric == 'acc':
                plt.title('Performance - Accuracy', size=15, color='black')
                plt.ylabel('Accuracy', size=15, color='black')
                if metric == 'acc':
                    metric = 'train_acc'

            elif metric == 'loss' or metric == 'val_loss':
                plt.title('Performance - Loss', size=15, color='black')
                plt.ylabel('Loss', size=15, color='black')
                if metric == 'loss':
                    metric = 'train_loss'

            file_path_img = f"C:/Users/toanb/Dropbox/Apps/Overleaf/Master thesis/result_images/BOX_{metric}_engine2.png"
            ax = sns.boxplot(data=values_only)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, color='black',
                               rotation=40, ha="right",
                               )
            plt.tight_layout()
            rc = {'axes.labelsize': 40, 'font.size': 40, 'legend.fontsize': 40, 'axes.titlesize': 40}
            sns.set(rc=rc)
            # plt.savefig(file_path_img)
            plt.show()
            print(f"image saved: {file_path_img}")
            # plt.close()
            continue

        plt.legend(prop={'size': 11})
        if metric == 'acc':
            plt.title('Training Accuracy', size=15)
            plt.ylabel('Accuracy', size=15)
            metric = 'train_acc'
        elif metric == 'val_acc':
            plt.title('Validation - Accuracy', size=15)
            plt.ylabel('Accuracy', size=15)
        elif metric == 'loss':
            plt.title('Training Loss', size=15)
            plt.ylabel('Loss', size=15)
            metric = 'train_loss'
        elif metric == 'val_loss':
            plt.title('Validation - Loss', size=15)
            plt.ylabel('Loss', size=15)
        plt.xlabel('Epoch', size=15)


        file_path_img = f"C:/Users/toanb/Dropbox/Apps/Overleaf/Master thesis/result_images/{metric}_adding_aug.png"

        plt.savefig(file_path_img)
        plt.show()
        print(f"image saved: {file_path_img}")
        plt.close()

def plot_signals(signals, channel='stereo'):
    """
    Plot time signals
    :param signals:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20, 5))

    fig.suptitle(f'Time Series - {channel}', size=16)

    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i], fontsize=30)
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
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms CLEAN', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(True)
            axes[x, y].get_yaxis().set_visible(True)
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


def plot_spectrogram_with_cool_axes(signal, sr, spectrogram, c):
    """
    so cool
    :param signal:
    :param sr:
    :param spectrogram:
    :return:
    """
    fig, ax = plt.subplots(1, 2, sharex=False,
                           sharey=False,
                           figsize=(6, 5),
                           )

    # plt.gca().set_aspect('equal', adjustable='box')
    # ax[0] = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    # ax[0] = plt.subplot(211)
    # ax[1] = plt.subplot(212)
    plt.grid(False)
    steps = np.linspace(0, 4, len(signal))
    ax[0].set_title(fix_title(c))
    ax[0].axes.set_xlim([0, 4])
    ax[0].plot(steps, signal, color='black')
    plt.grid(False)
    steps = np.linspace(0, 4, len(signal))
    ax[0].set_title(fix_title(c))
    ax[0].axes.set_xlim([0, 4])
    ax[0].plot(steps, signal, color='black')

    ax[0].set_ylabel('Amplitude', color='black')
    ax[0].set_xlabel('Time [seconds]', color='black')
    # ax[0].get_xaxis().set_visible(False)
    ax[1].set_ylabel('Frequency [Hz]', color='black')
    ax[1].set_xlabel('Time [seconds]', color='black')

    ax[1].specgram(signal, Fs=sr,
                                 # NFFT=512, window=np.hamming(512), axes=ax[1]
                                  )
    # ax[2].set_ylabel('Filters', color='black')
    # ax[2].set_xlabel('Time [Frames]', color='black')
    # busk1 = ax[2].imshow(spectrogram[0])
    # ax[3].set_ylabel('Coeffs', color='black')
    # ax[3].set_xlabel('Time [Frames]', color='black')
    # busk = ax[3].imshow(spectrogram[1])
    #
    # # fig.colorbar(busk1).set_label('Power [dB]')
    # plt.tight_layout()
    plt.show()


def fix_title(c):
    """
    Converts the lower case class of UrbanSound to Upper Case with space instead of underbar
    :param c:
    :return:
    """
    if c == 'children_playing':
        return 'Children Playing'
    elif c == 'gun_shot':
        return 'Gun Shot'
    elif c == 'car_horn':
        return 'Car Horn'
    elif c == 'jackhammer':
        return 'Jackhammer'
    elif c == 'drilling':
        return 'Drilling'
    elif c == 'siren':
        return 'Siren'
    elif c == 'dog_bark':
        return 'Dog Bark'
    elif c == 'street_music':
        return 'Street Music'
    elif c == 'air_conditioner':
        return 'Air Conditioner'
    elif c == 'engine_idling':
        return 'Engine Idling'


def plot_mfccs(mfccs):
    """
    Plots the mel filter cepstrum coefficients
    :param mfccs:
    :return:
    """
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(4):
        for y in range(1):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i], cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_class_distribution(class_dist, labels=None):
    """
    Plots a cake diagram of the distribution of the various classes
    of the UrbanSound dataset
    :param df:
    :return:
    """
    # Plot the distribution
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    if labels is None:
        ax.pie(class_dist, labels=class_dist.index, shadow=False, autopct='%1.1f%%', startangle=90)
    else:
        # ax.pie(class_dist, labels=list(np.unique(labels['label'])), shadow=False, autopct='%1.1f%%', startangle=90)
        ax.pie(class_dist, labels=labels, shadow=False, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.show()
# df.reset_index(inplace=True)
