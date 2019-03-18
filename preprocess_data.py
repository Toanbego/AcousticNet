"""
Author - Torstein Gombos
Date - 08.01.2019

Script that is i used to plot and analyze the audio signals.
Various tests can be run in regards to sampling rates, feature extraction and etc.

The script is also used to clean and downsample signals so they are of equal length and size.
"""


import pandas as pd
import argparse
import resampy

from tqdm import tqdm
import librosa
from scipy.io import wavfile
import soundfile as sf
from python_speech_features import mfcc, logfbank, fbank, sigproc, delta, get_filterbanks
from python_speech_features import sigproc
import numpy as np
from scipy.signal import spectral, spectrogram, periodogram, get_window
import sys
from matplotlib import pyplot as plt
import configparser
from keras.utils import to_categorical
import plotting_functions


# Parse the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")

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

    parser.add_argument('--mask', '-m', type=str,
                        help='activate to use thresholding when downsampling the files',
                        default='False')

    args = parser.parse_args()
    return args


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
        signal, rate = sf.read(f'../Datasets/audio/new_test/fold{fold}/{f}')
        df.at[f, 'length'] = signal.shape[0] / rate

    df.to_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length_NewTest.csv')
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

        left = channels[0]
        right = channels[1]

    elif y.shape[1] > y.shape[0]:
        # y = np.reshape(y, (y.shape[1], y.shape[0]))
        channels = np.array(y)

        left = channels[0]
        right = channels[1]

    return left, right


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j, i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j, i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def extract_features(signal, rate, win_func=lambda x: np.hamming(x, ), clean=False, fft=False, filterbank=False,
                     mffc=False, dynamic_threshold=False):
    """
    Reads a signal and calculates the fft, filter bank and mfcc.
    Always return the signal.
    :param win_func:
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
            mask = np.array(preprocessing.envelope(signal, rate, threshold=threshold))

            # If threshold is to high, reduce the threshold
            while mask.sum() < (len(signal)*0.8):
                print("Threshold is to high. Reducing threshold")
                threshold *= 0.95
                mask = np.array(preprocessing.envelope(signal, rate, threshold=threshold))

        else:
            mask = preprocessing.envelope(signal, rate, threshold=threshold, dynamic_threshold=False)

        # Mask the signal
        signal = signal[mask]

    list_of_returns.append(signal)

    # Calculate fourier transform
    if fft is True:
        fft = calc_fft(signal, rate)
        list_of_returns.append(fft)

    # Find filter bank coefficients
    if filterbank is True:
        bank = sbank(signal, rate)[0]
        bank = np.log(bank).T
        list_of_returns.append(bank)

    # Find mel frequency
    if mffc is True:
        mel = mfcc(signal, rate, winfunc=win_func).T
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


def sbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.95,
          winfunc=lambda x: np.ones((x,))):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, 0)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energys = np.sum(pspec, 1)  # this stores the total energy in each frame
    energys = np.where(energys == 0, np.finfo(float).eps, energys)  # if energy is zero, we get problems with log



    highfreq = highfreq or samplerate / 2
    signal_hat = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal_hat, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = np.sum(pspec, 1)  # this stores the total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    plt.title('Power Spectral Density - Jackhammer', size=20)
    plt.xlabel('Frequency (HZ')
    plt.ylabel('Power (dB)')
    plt.plot(energys, label='No Pre-Emphasis', color='r')
    plt.plot(energy, label='With Pre-Emphasis')
    plt.legend()
    plt.show()
    exit()


    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy, fb


class PreprocessData:

    def __init__(self, df):
        # DataFrame
        self.length_files = df.loc[df['length'] > config['preprocessing'].getfloat('signal_length')]
        self.df = df.loc[df['length'] >= 3.5]
        self.df = self.df.reset_index()
        # Find classes and create class distribution
        self.classes = [
            # 'children_playing',
            # 'air_conditioner',
            # 'engine_idling',
            'siren',
            # 'street_music',
            'drilling',
            'jackhammer',
            # 'dog_bark',
            # 'gun_shot',
            # 'car_horn'
        ]

        # Audio parameters
        self.network_type = config['model']['net']
        self.save_file = config['preprocessing']['save_file']
        self.randomize_rolls = config['preprocessing'].getboolean('randomize_roll')

        self.feature = config['preprocessing']['feature']
        self.n_filt = config['preprocessing'].getint('n_filt')
        self.n_feat = config['preprocessing'].getint('n_feat')
        self.n_fft = config['preprocessing'].getint('n_fft')
        self.rate = config['preprocessing'].getint('rate')
        self.step_length = config['preprocessing'].getfloat('step_size')
        self.sample_length = int(self.rate * self.step_length)

        self.activate_threshold = config['preprocessing'].getboolean('activate_threshold')
        self.threshold = config['preprocessing'].getfloat('threshold')
        self.n_samples = config['preprocessing'].getint('n_samples')
        self.random_extraction = config['preprocessing'].getboolean('random_extraction')

        # Training parameters
        self.n_training_samples = self.find_samples_per_epoch(start_fold=1, end_fold=9)
        self.n_validation_samples = self.find_samples_per_epoch(start_fold=9, end_fold=10)
        self.n_testing_samples = self.find_samples_per_epoch(start_fold=10, end_fold=11)
        self.training_seed = np.random.seed()
        self.validation_seed = np.random.seed()

        self.class_dist = [self.n_training_samples[classes] / sum(self.n_training_samples.values())
                           for classes in self.classes]
        self.prob_dist = pd.Series(self.class_dist, self.classes)
        self.validation_fold = None

    def build_feature_from_signal(self, sample, rate, file_path, feature_to_extract='mfcc',
                                  activate_threshold=False, seed=None):
        """
        Reads the signal from the file path. Then build mffc features that is returned.
        If the signal is stereo, the signal will be split up and only the first channel is used.
        Later implementations should consist of using both channels, and being able to select other features than mfccs
        :param sample:
        :param rate:
        :param seed: None by default
        :param feature_to_extract: Choose what feature to extract. Current alternatives: 'mffc', 'fbank' and 'logfbank'
        :param file_path: File path to audio file
        :param activate_threshold: A lower boundary to filter out weak signals
        :return:
        """
        if self.random_extraction is True:
            # Read file
            sample, rate = sf.read(file_path)
            sample = self.make_signal_mono(sample)

            # Choose a window of the signal to use for the sample
            try:
                sample = self.select_random_audio_clip(sample, seed)

            except ValueError:
                print("audio file to small. Skip and move to next")
                return 'move to next file'

        # Perform filtering with a threshold on the time signal
        if activate_threshold is True:
            mask = self.envelope(sample, rate, self.threshold)
            sample = sample[mask]

        # Extracts the mel frequency cepstrum coefficients
        if feature_to_extract == 'mfcc':
            sample_hat = mfcc(sample, rate,
                              numcep=self.n_feat,
                              nfilt=self.n_filt,
                              nfft=self.n_fft).T

        # Extract the log mel frequency filter banks
        elif feature_to_extract == 'logfbank':
            # TODO: Fiks detta
            Sxx = spectrogram(sample, rate, noverlap=240, nfft=512, window=get_window('hamming', 400, 512))

            sample_hat = logfbank(sample, rate,
                                  nfilt=self.n_filt,
                                  nfft=self.n_fft).T

        # Extract the mel frequency filter banks
        elif feature_to_extract == 'fbank':
            sample_hat = fbank(sample, rate,
                               nfilt=self.n_filt,
                               nfft=self.n_fft)[0].T

        elif feature_to_extract == 'spectogram':
            f, t, Sxx = spectrogram(sample, rate, noverlap=240,
                                           nfft=512,
                                           window=get_window('hamming', 400, 512))
            sample_hat = np.where(Sxx == 0, np.finfo(float).eps, Sxx)
            sample_hat = np.log(sample_hat)

        else:
            raise ValueError('Please choose an existing feature in the config.ini file: mfcc, logfbank, '
                             'spectogram or fbank')
        return sample_hat

    def select_random_audio_clip(self, sample, seed=None):
        """
        Selects a part of the audio file at random. length of clip is defined by self.step
        :return:
        """
        if seed is None:
            rand_index = np.random.randint(0, sample.shape[0] - self.sample_length)
        else:
            np.random.seed(seed)
            rand_index = np.random.randint(0, sample.shape[0] - self.sample_length)
        return sample[rand_index:rand_index + self.sample_length]

    def find_samples_per_epoch(self, start_fold=1, end_fold=9):
        """
        Finds all the samples that exist in the data set for a particular class.
        It then checks the defined step length and checks how many possible samples that
        that can be extracted for the class.
        :param folds: what folds to check
        :return:
        """
        samples_dict = {}
        for fold in range(start_fold, end_fold):
            # Find all the files in the fold
            files_in_fold = self.df[self.df.fold == fold]

            for classes in self.classes:
                # Find all the matches of that class in the fold
                files_with_class = files_in_fold[self.df.label == classes]

                # Add up the samples
                if classes in samples_dict.keys():
                    samples_dict[classes] += int(files_with_class['length'].sum()/self.step_length)
                else:
                    samples_dict[classes] = int(files_with_class['length'].sum()/self.step_length)
        return samples_dict

    def preprocess_dataset_generator(self, mode='training'):
        """
        Pre-processes a batch of data which is yielded for every function call. Size is defined by batch size
        is defined the config.ini file. If mode = 'training', the first call will make a batch from the first
        available fold of the training folds. Next call will be from the next available fold. When the end
        is reached it will start from the first fold again.
        :return:
        """

        _min, _max = float('inf'), -float('inf')    # Initialize min and max for x
        folds = np.unique(self.df['fold'])    # The folds to loop through

        # Separate folds into training, validation and split
        folds = np.roll(folds, folds[-1] - self.fold)
        training_folds = folds[:-1]
        validation_folds = folds[-1]
        testing_folds = folds[-1]

        self.validation_fold = self.fold

        # Choose fold to start from
        if mode == 'training':
            fold = 1
            seed = self.training_seed
        elif mode == 'validation':
            fold = validation_folds
            seed = self.validation_seed
        elif mode == 'testing':
            fold = testing_folds
            seed = 42

        # Build feature samples
        while True:
            x, y = [], []  # Set up lists

            # Find the files in the current fold
            files_in_fold = self.df.loc[self.df.fold == fold]

            # Loop through the files in the fold and create a batch
            for n in range(self.batch_size):

                # Pick a random class from the probability distribution and then a random file with that class
                rand_class = np.random.choice(self.classes, p=self.prob_dist)
                file = np.random.choice(files_in_fold[self.df.label == rand_class].slice_file_name)
                file_path = f'../Datasets/audio/lengthy_audio/fold{fold}/{file}'

                # Extract feature from signal
                x_sample = self.build_feature_from_signal(None, None, file_path,
                                                          feature_to_extract=self.feature,
                                                          activate_threshold=self.activate_threshold,
                                                          seed=seed)

                # If the flag is set, it means it could not process current file and it moves to next.
                if x_sample == 'move to next file':
                    print(rand_class)
                    continue

                # Update min and max values
                _min = min(np.amin(x_sample), _min)
                _max = max(np.amax(x_sample), _max)

                # Create batch set with corresponding labels
                x.append(x_sample)
                y.append(self.classes.index(rand_class))

                n += 1

            # Normalize X and y and reshape the features
            y = np.array(y)
            x = np.array(x)
            x = (x - _min) / (_max - _min)                          # Normalize x and y
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Reshape all to same size

            # One-hot encode the labels and append to fold
            y = to_categorical(y, num_classes=len(self.classes))    # One hot encoding

            # Reset fold number when reaching the end
            if mode == 'training':
                fold += 1

                if fold > len(training_folds):
                    fold = 1

            # Return data and labels
            yield x, y

    def preprocess_dataset(self):
        """
        Builds and shapes the data according to the mode chosen.

        UrbanSounds homepage encourages not to shuffle the data, and mix the folds together.
        The sample is converted to a spectrogram and appended into an list of X and Y
        and is to be used as features to train the model.
        A normal sample length could be 1/10th of a second, but one could
        always try more or less.
        :return:
        """
        fold_list_x = []
        fold_list_y = []
        files_with_label = {}

        _min, _max = float('inf'), -float('inf')    # Initialize min and max for x
        folds = list(np.unique(self.df['fold']))    # The folds to loop through
        self.df = self.df.reset_index()

        # Build feature samples
        for idx,  fold in enumerate(folds):
            x, y = [], []  # Set up lists
            print(f'\nExtracting data from fold{fold}')

            # Get the filenames that exists in that fold
            files_in_fold = self.df.loc[self.df.fold == fold]
            for classes in self.classes:
                files_with_label[classes] = files_in_fold[self.df.label == classes].slice_file_name

                # Loop through the files in the fold
                for file in tqdm(files_with_label[classes]):

                    # Read file
                    file_path = f'../Datasets/audio/lengthy_audio/fold{fold}/{file}'
                    sample, rate = sf.read(file_path)                                   # Read the file
                    sample = self.make_signal_mono(sample)                              # Make signal mono
                    length = float(self.df[self.df.slice_file_name == file].length)     # Find length of signal
                    samples_from_signal = int(length / self.step_length)                # Possible samples to get

                    # Loop through the signal
                    for i in range(samples_from_signal):
                        sample_length = self.step_length * rate
                        # Extract feature from signal
                        x_sample = self.build_feature_from_signal(sample[int(sample_length*i):int(sample_length*(i+1))],
                                                                  rate,
                                                                  None,
                                                                  feature_to_extract=self.feature,
                                                                  activate_threshold=self.activate_threshold)

                        # Update min and max
                        _min = min(np.amin(x_sample), _min)
                        _max = max(np.amax(x_sample), _max)
                        x.append(x_sample)
                        y.append(self.classes.index(classes))

            # Normalize X and y and reshape the features
            y = np.array(y)
            x = np.array(x)
            x = (x - _min) / (_max - _min)                          # Normalize x and y
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Reshape all to same size

            # One-hot encode the labels and append to fold
            y = to_categorical(y, num_classes=len(self.classes))    # One hot encoding
            fold_list_x.append(x)
            fold_list_y.append(y)

        # Save the features in a .npz file
        np.savez(self.save_file, x=fold_list_x, y=fold_list_y)

    def preprocessing_dataset_randomly(self):
        """
        Builds and shapes the data according to the mode chosen.

        UrbanSounds homepage encourages not to shuffle the data, and mix the folds together.
        The sample is converted to a spectrogram and appended into an list of X and Y
        and is to be used as features to train the model.
        A normal sample length could be 1/10th of a second, but one could
        always try more or less.
        :return:
        """
        fold_list_x = []
        fold_list_y = []

        _min, _max = float('inf'), -float('inf')  # Initialize min and max for x
        folds = list(np.unique(self.df['fold']))  # Initialize the folds to loop through
        self.df = self.df.reset_index()           # Reset the data frame index

        # Build feature samples
        for idx, fold in enumerate(folds):
            x, y = [], []  # Set up lists
            print(f'\nExtracting data from fold{fold}')

            # Get the filenames that exists in that fold
            files_in_fold = self.df.loc[self.df.fold == fold]  # Get the filenames that exists in that fold

            # Multiply the number of sample by 1.2 to add some overlapping samples
            samples_per_fold = sum(self.find_samples_per_epoch(fold, fold + 1).values())*1.2
            print(f'Extracting {samples_per_fold} samples for fold {fold}:\n')

            # Loop through the files in the fold
            for _ in tqdm(range(samples_per_fold)):

                # Pick a random class from the probability distribution and then a random file with that class
                rand_class = np.random.choice(self.classes, p=self.prob_dist)
                file = np.random.choice(files_in_fold[self.df.label == rand_class].slice_file_name)
                file_path = f'../Datasets/audio/lengthy_audio/fold{fold}/{file}'

                # Extract feature from signal
                x_sample = self.build_feature_from_signal(None, None, file_path,
                                                          feature_to_extract=self.feature,
                                                          activate_threshold=self.activate_threshold)

                # If the flag is set, it means it could not process current file and it moves to next.
                if x_sample == 'move to next file':
                    print(rand_class)
                    continue

                # Update min and max
                _min = min(np.amin(x_sample), _min)
                _max = max(np.amax(x_sample), _max)
                x.append(x_sample)
                y.append(self.classes.index(rand_class))

            # Normalize X and y and reshape the features
            y = np.array(y)
            x = np.array(x)
            x = (x - _min) / (_max - _min)  # Normalize x and y
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)  # Reshape all to same size

            # One-hot encode the labels and append to fold
            y = to_categorical(y, num_classes=len(self.classes))  # One hot encoding
            fold_list_x.append(x)
            fold_list_y.append(y)

        # Save the features in a .npz file
        np.savez(self.save_file, x=fold_list_x, y=fold_list_y)

    def envelope(self, y, rate, threshold):
        """
        Function that helps remove redundant information from time series
        signals.
        :param y: signal
        :param rate: sampling rate
        :param threshold: magnitude threshold
        :param dynamic_threshold
        :return: Boolean array mask
        """

        # Checks one column of the signal dataframe
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate / 15), min_periods=1, win_type='hamming', center=True).mean()

        m = np.greater(y_mean, threshold)

        return m

    def make_signal_mono(self, y):
        """
        If a signal is stereo, average out the channels and make the signal mono
        :return:
        """
        if len(y.shape) > 1:
            y = y.reshape((-1, y.shape[1])).T
            y = np.mean(y, axis=0)
        return y

    def resample_signal(self, y, orig_sr, target_sr):
        """
        Resamples a signal from original samplerate to target samplerate
        :return:
        """

        if orig_sr == target_sr:
            return y

        # 1 - step
        ratio = float(target_sr) / orig_sr
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
        return np.ascontiguousarray(y_hat), sr

    def downsample_all_signals(self, high_freq=16000):
        """
        Loops through all the sound files and applies a high pass filter
        :return:
        """
        # Store in clean directory
        for wav_file in tqdm(self.df.slice_file_name):

            # Find filename and filepath
            fold = self.df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
            file_name = f'../Datasets/audio/original/fold{fold}/{wav_file}'

            # Read file, monotize if stereo and resample
            signal, sr = sf.read(file_name)
            signal = preprocessing.make_signal_mono(signal)
            signal_hat, sr = preprocessing.resample_signal(signal, orig_sr=sr, target_sr=high_freq)

            # Apply thresholding if true
            if args.mask == 'True':
                mask = preprocessing.envelope(signal_hat, sr, threshold=0.005)

            # Write to file
            wavfile.write(filename=f'../Datasets/audio/lengthy_audio/fold{fold}/{wav_file}',
                          rate=high_freq,
                          data=signal_hat)

    def calculate_power_spectrum(self, signal, samplerate=16000, winlen=0.025, winstep=0.01,
                                 nfilt=26, nfft=512, preemph=0.97,
                                 winfunc=lambda x: np.ones((x,))):
        """
        Frames a signal and calculates the power spectrum
        :return:
        """
        y = sigproc.preemphasis(signal, preemph)
        frames = sigproc.framesig(y, winlen * samplerate, winstep * samplerate, winfunc)
        pow_spec = sigproc.powspec(frames, nfft)

        return pow_spec


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

    preprocessing = PreprocessData(df)

    # Fetch the classes from the CSV file
    classes = list(np.unique(df.label))

    # Dicts for things we want to store
    signals = {}
    mfccs = {}
    filterbank = {}
    f_bank = {}
    fft = {}
    mfccs_low = {}
    filterbank_low = {}
    mfccs_low_no_mask = {}
    logfbank_low_no_mask = {}
    Sxx = {}
    # # Loop through folds and calculate spectrogram and plot data
    #
    # # banks = get_filterbanks(10, 512, 16000, lowfreq=300)
    # # plt.figure(figsize=(20, 5))
    # # for b in banks:
    # #     plt.plot(b)
    # # plt.show()
    c = 'jackhammer'
    wav_file = df[df.label == c].iloc[0, 0]
    fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
    target_sr = 16000
    signal, sr = sf.read(f'../Datasets/audio/original/fold{fold}/{wav_file}')
    signal = preprocessing.make_signal_mono(signal)
    signal, sr = preprocessing.resample_signal(signal, orig_sr=sr, target_sr=target_sr)

    sbank(signal, sr, preemph=0.9)

    mask = preprocessing.envelope(signal, sr, 0.027)

    fourier_mask = calc_fft(signal[mask], sr)
    fourier = calc_fft(signal, sr)
    plt.title("Jackhammer", size=20)
    plt.xlabel('Frequency (Hz)', size=10)
    plt.ylabel('Magnitude', size=10)
    plt.plot(fourier_mask[1], fourier_mask[0], color='r', label='After Thresholding')
    plt.plot(fourier[1], fourier[0], label='Before Thresholding')
    plt.legend()
    plt.show()
    exit()

    # f, t, Sxx = spectrogram(signal, target_sr, noverlap=240, nfft=512, window=get_window('hamming', 400, 512))
    # pow_spec = preprocessing.calculate_power_spectrum(signal, target_sr, winfunc=lambda x: np.hamming(x, )).T
    # # period = periodogram(signal, target_sr, window=get_window('hanning', 400, 512), nfft=512)
    # logo = np.log(sbank(signal, target_sr, winfunc=lambda x: np.hamming(x, ))[0]).T
    #
    # filterbanks = get_filterbanks(26, 512, 16000).T
    #
    # logfbank = np.log(np.dot(Sxx.T, filterbanks).T)
    #
    # plt.title("specto")
    # plt.pcolormesh(t, f, np.log(Sxx))
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    #
    # plt.title("logo")
    # plt.imshow(logfbank, cmap='hot', interpolation='nearest')
    # plt.show()
    #
    # # plt.figure(figsize=(20, 15))
    # # plt.imshow(np.log(spectog[2]))
    # # plt.show()
    # plt.title("pow_spec")
    # plt.imshow(np.log(pow_spec))
    # plt.show()
    #
    #
    # exit()

    classes = ['siren', 'jackhammer']

    for c in classes:
        # Get the file name and the fold it exists in from the dataframe
        wav_file = df[df.label == c].iloc[0, 0]
        fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
        target_sr = 16000
        # Read signal and add it to dict. UrbanSound uses stereo which is made mono
        signal, sr = sf.read(f'../Datasets/audio/original/fold{fold}/{wav_file}')
        signal = preprocessing.make_signal_mono(signal)
        signal, sr = preprocessing.resample_signal(signal, orig_sr=sr, target_sr=target_sr)

        signals[c], fft[c], logfbank_low_no_mask[c], mfccs_low_no_mask[c] = extract_features(signal, target_sr,
                                                                               win_func=lambda x: np.hamming(x, ),
                                                                               clean=False,
                                                                               fft=True,
                                                                               filterbank=True,
                                                                               mffc=True,
                                                                               dynamic_threshold=False)

        # f, t, Sxx[c] = spectrogram(signal, target_sr, noverlap=240, nfft=512, window=get_window('hamming', 400, 512))
        # mask = preprocessing.envelope(signal, target_sr, 0.006)
        # signal_hat = signal[mask]
        #
        # mfccs_low[c] = mfcc(signal_hat, target_sr, lowfreq=0, highfreq=8000, winfunc=lambda x: np.hamming(x, )).T
        # feat = sbank(signal_hat, target_sr, winfunc=lambda x: np.hamming(x, ), preemph=0.97)[0]
        # filterbank_low[c] = np.log(feat).T

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle('Time Signal', size=20, y=1)
    time_scale = np.linspace(0, 4, len(signals['siren']))

    axes[0].set_title("Siren", size=20)
    axes[0].plot(time_scale, signals['siren'])
    axes[0].set_ylabel('Amplitude', size=20)
    axes[0].set_xlabel('Time [sec]', size=20)

    axes[1].set_title("Jackhammer", size=20)
    axes[1].plot(time_scale, signals['jackhammer'])
    axes[1].set_ylabel('Amplitude', size=20)
    axes[1].set_xlabel('Time [sec]', size=20)
    plt.show()
    exit()

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('Siren', size=20)
    time_scale = np.linspace(0, 4, len(signals[c]))
    xcoords = np.linspace(0, 4, 16)
    axes[0, 0].set_title("Framed Signal", size=20)
    axes[0, 0].plot(time_scale, signals[c])
    axes[0, 0].set_ylabel('Amplitude', size=20)
    axes[0, 0].set_xlabel('Time [sec]', size=20)
    for xc in xcoords:
        axes[0, 0].axvline(x=xc, color='black', linestyle='--')

    axes[0, 1].set_title("STFT", size=20)
    axes[0, 1].plot(fft[c][1], fft[c][0])
    axes[0, 1].set_ylabel('Magnitude [dB]', size=20)
    axes[0, 1].set_xlabel('Frequency [Hz]', size=20)

    axes[1, 0].set_title("Power Spectral Features", size=20)
    axes[1, 0].pcolormesh(t, f, Sxx[c], cmap='hot')
    axes[1, 0].set_ylabel('Frequency [Hz]', size=20)
    axes[1, 0].set_xlabel('Time [sec]', size=20)

    axes[1, 1].set_title("Log Power Spectral Features", size=20)
    axes[1, 1].pcolormesh(t, f, np.log(Sxx[c]), cmap='hot')
    axes[1, 1].set_ylabel('Frequency [Hz]', size=20)
    axes[1, 1].set_xlabel('Time [sec]', size=20)
    plt.show()




        # axes[0, 0].set_title("Low_freq_mask")
        # axes[0, 0].imshow(mfccs_low[c], cmap='hot', interpolation='nearest')
        #
        # axes[0, 1].set_title("Low_freq")
        # axes[0, 1].imshow(mfccs_low_no_mask[c], cmap='hot', interpolation='nearest')
        #
        # axes[1, 0].set_title("LowFreq_mask")
        # axes[1, 0].imshow(filterbank_low[c], cmap='hot', interpolation='nearest')
        #
        # axes[1, 1].set_title("Low_freq")
        # axes[1, 1].imshow(logfbank_low_no_mask[c], cmap='hot', interpolation='nearest')

    plt.show()

    # Clean and downsample the wav files
    if args.clean_files == 'True':
        preprocessing.hi_pass_filter(8000)










