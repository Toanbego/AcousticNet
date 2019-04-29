"""
Author - Torstein Gombos
Date - 08.01.2019

Preprocessing script for all your preprocessing needs
"""


# Signal processing libraries
import resampy
import librosa
import muda
import jams
from scipy.io import wavfile
import soundfile as sf
import pyrubberband as pyrb
from python_speech_features import mfcc, logfbank, fbank, sigproc, delta, get_filterbanks
from scipy.signal import spectrogram, get_window
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import math

from PIL import Image
import pandas as pd
from tqdm import tqdm

# Standard libraries
import numpy as np
from matplotlib import pyplot as plt
import configparser
from keras.utils import to_categorical
# Parse the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")
import os
import re

import plotting_functions

def i_screwed_up_and_need_to_rename_all_my_files(df):
    """
    Rename all the files god dammit
    1. Find the pattern than needs to move
    2. Move the pattern
    3. Rename the file
    4. Change the name of the file in the csv file

    :return:
    """
    exit()
    for fold in os.listdir("../Datasets/audio/augmented"):
        for name in os.listdir(f"../Datasets/audio/augmented/{fold}"):
            # Find the pattern
            pattern = re.findall('wav(_.+)', name)

            if not pattern:
                continue

            if pattern:
                # Create the new pattern
                org_name = re.findall('(.+).wav', name)
                new_name = org_name[0] + pattern[0] + '.wav'

                # Change the name of the file
                os.rename(f'../Datasets/audio/augmented/{fold}/{name}',
                          f'../Datasets/audio/augmented/{fold}/{new_name}')

    exit()

    # Set column to index to loop through it faster
    df.set_index('slice_file_name', inplace=True)

    for name in tqdm(df.index):
        # Find the pattern
        pattern = re.findall('wav(_.+)', name)

        if not pattern:
            continue

        if pattern:
            # Create the new pattern
            org_name = re.findall('(.+).wav', name)
            new_name = org_name[0]+pattern[0]+'.wav'

            # Change name of csv file
            df.rename(index={name: new_name}, inplace=True)

            # Change the name of the file
            fold = df.loc[df.index == new_name, 'fold'].iloc[0]
            os.rename(f'../Datasets/audio/augmented/fold{fold}/{name}',
                      f'../Datasets/audio/augmented/fold{fold}/{new_name}')


    df = df.reset_index()
    df.to_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length_augmented.csv')
    exit()


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
    for f in tqdm(df.index):
        signal, rate = sf.read(f'../Datasets/audio/augmented/fold{fold}/{f}')
        df.at[f, 'length'] = signal.shape[0] / rate
        # param = re.findall('_(.*)_shift_(.*).wav', f)
        # if param:
        #     param = list(param[0])
        #     df.at[f, 'param'] = param[1]
        #     df.at[f, 'augmentation'] = param[0]
        # if not param:
        #     df.at[f, 'param'] = float('NaN')

    df.to_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')
    exit()
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


def cwt(data, wavelet_name, sampling_frequency=1., voices_per_octave=10):
    """
    cwt(data, scales, wavelet)
    One dimensional Continuous Wavelet Transform.
    Parameters
    ----------
    data : array_like
        Input signal
    wavelet_name : Wavelet object or name
        Wavelet to use. Currently, only the Morlet wavelet is supported ('morl').
    sampling_frequency : float
        Sampling frequency for frequencies output (optional)
    precision : int
        Presicion of the signal
    Returns
    -------
    coefs : array_like
        Continous wavelet transform of the input signal for the given scales
        and wavelet
    frequencies : array_like
        if the unit of sampling period are seconds and given, than frequencies
        are in hertz. Otherwise Sampling period of 1 is assumed.
    Notes
    -----
    Size of coefficients arrays is automatically calculated given the wavelet and the data length. Currently, only the
    Morlet wavelet is supported.
    Examples
    --------
    fs = 1e3
    t = np.linspace(0, 1, fs+1, endpoint=True)
    x = np.cos(2*np.pi*32*t) * np.logical_and(t >= 0.1, t < 0.3) + np.sin(2*np.pi*64*t) * (t > 0.7)
    wgnNoise = 0.05 * np.random.standard_normal(t.shape)
    x += wgnNoise
    c, f = cwt.cwt(x, 'morl', sampling_frequency=fs, plot_scalogram=True)
    """

    # Currently only supported for Morlet wavelets
    if wavelet_name == 'morl':
        data -= np.mean(data)
        n_orig = data.size
        nv = voices_per_octave
        ds = 1 / nv
        fs = sampling_frequency
        dt = 1 / fs

        # Pad data symmetrically
        padvalue = n_orig // 2
        x = np.concatenate((np.flipud(data[0:padvalue]), data, np.flipud(data[-padvalue:])))
        n = x.size

        # Define scales
        _, _, wavscales = cwt_filterbank(wavelet_name, n_orig, ds)

        wavscales = wavscales[:]
        num_scales = wavscales.size

        # Frequency vector sampling the Fourier transform of the wavelet
        omega = np.arange(1, math.floor(n / 2) + 1, dtype=np.float64)
        omega *= (2 * np.pi) / n
        omega = np.concatenate((np.array([0]), omega, -omega[np.arange(math.floor((n - 1) / 2), 0, -1, dtype=int) - 1]))

        # Compute FFT of the (padded) time series
        f = np.fft.fft(x)

        # Loop through all the scales and compute wavelet Fourier transform
        psift, freq = waveft(wavelet_name, omega, wavscales)


        # Inverse transform to obtain the wavelet coefficients.
        cwtcfs = np.fft.ifft(np.kron(np.ones([num_scales, 1]), f) * psift)
        cfs = cwtcfs[:, padvalue:padvalue + n_orig]
        freq = freq * fs

        return cfs, freq
    else:
        raise Exception


def cwt_filterbank(wavelet, n, ds):
    """
    getDefaultScales(wavelet, n, ds)
    Calculate default scales given a wavelet and a signal length.
    Parameters
    ----------
    wavelet : string
        Name of wavelet
    n : int
        Number of samples in a given signal
    ds : float
        Scale resolution (inverse of number of voices in octave)
    Returns
    -------
    s0 : int
        Smallest useful scale
    ds : float
        Scale resolution (inverse of number of voices in octave). Here for legacy reasons; implementing more wavelets
        will need this output.
    scales : array_like
        Array containing default scales.
    """
    wname = wavelet
    nv = 1 / ds

    if wname == 'morl':

        # Smallest useful scale (default 2 for Morlet)
        s0 = 2

        # Determine longest useful scale for wavelet
        max_scale = n // (np.sqrt(2) * s0)
        if max_scale <= 1:
            max_scale = n // 2
        max_scale = np.floor(nv * np.log2(max_scale))
        a0 = 2 ** ds
        scales = s0 * a0 ** np.arange(0, max_scale + 1)
    else:
        raise Exception

    return s0, ds, scales


def waveft(wavelet, omega, scales):
    """
    waveft(wavelet, omega, scales)
    Computes the Fourier transform of a wavelet at certain scales.
    Parameters
    ----------
    wavelet : string
        Name of wavelet
    omega : array_like
        Array containing frequency values in Hz at which the transform is evaluated.
    scales : array_like
        Vector containing the scales used for the wavelet analysis.
    Returns
    -------
    wft : array_like
        (num_scales x num_freq) Array containing the wavelet Fourier transform
    freq : array_like
        Array containing frequency values
    """
    wname = wavelet
    num_freq = omega.size
    num_scales = scales.size
    wft = np.zeros([num_scales, num_freq])

    if wname == 'morl':
        gC = 6
        mul = 2
        for jj, scale in enumerate(scales):
            expnt = -(scale * omega - gC) ** 2 / 2 * (omega > 0)
            wft[jj, ] = mul * np.exp(expnt) * (omega > 0)

        fourier_factor = gC / (2 * np.pi)
        frequencies = fourier_factor / scales

    else:
        raise Exception

    return wft, frequencies


def save_scalograms(preproc, df):
    """
    Extracts scalograms from audio files and saves as png.
    :param preproc:
    :param df:
    :return:
    """
    for wav_file in tqdm(df.slice_file_name[32009:]):

        # Find filename and filepath
        fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]

        file_name = f'../Datasets/audio/augmented/fold{fold}/{wav_file}'

        # Read file, monotize if stereo and resample
        msfb = preproc.extract_feature(file_name, 'msfb', random_extraction=False,
                                           activate_threshold=preprocessing.activate_envelope)
        plt.imshow(msfb, cmap='hot')
        plt.imsave('../Datasets/audio/msfb/fold' + str(fold) + '/' + str(wav_file) + '.jpeg', msfb)
        plt.close()

        mfcc = preproc.extract_feature(file_name, 'mfcc', random_extraction=False,
                                       activate_threshold=preprocessing.activate_envelope)
        plt.imshow(mfcc, cmap='hot')
        plt.imsave('../Datasets/audio/mfcc/fold' + str(fold) + '/' + str(wav_file) + '.jpeg', mfcc)
        plt.close()

        spectogram = preproc.extract_feature(file_name, 'spectogram', random_extraction=False,
                                             activate_threshold=preprocessing.activate_envelope)
        plt.imshow(spectogram, cmap='hot')
        plt.imsave('../Datasets/audio/spectogram/fold' + str(fold) + '/' + str(wav_file) + '.jpeg', spectogram)
        plt.close()
    exit()


class PreprocessData:

    def __init__(self, df):

        # Find classes and create class distribution
        self.classes = [
            # 'children_playing',
            'air_conditioner',
            'engine_idling',
            # 'siren',
            # 'street_music',
            'drilling',
            # 'jackhammer',
            # 'dog_bark',
            # 'gun_shot',
            # 'car_horn'
        ]

        # Augmentation parameters
        self.augmentations = config['augmentation']['augmentations']
        self.pitch_shift = config['augmentation'].getboolean('pitch_shift')
        self.time_shift = config['augmentation'].getboolean('time_shift')
        self.time_shift_param = list(map(float, config['augmentation']['time_shift_param'].split(',')))
        self.pitch_shift_param = list(map(float, config['augmentation']['pitch_shift_param'].split(',')))

        # Read and filter dataframes
        df_v = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')
        self.df, self.df_v = self.filter_dataframe(df, df_v)

        # Audio parameters
        self.randomize_rolls = config['preprocessing'].getboolean('randomize_roll')
        self.random_extraction = config['preprocessing'].getboolean('random_extraction')

        self.feature = config['preprocessing']['feature']
        self.n_filt = config['preprocessing'].getint('n_filt')
        self.n_feat = config['preprocessing'].getint('n_feat')
        self.n_fft = config['preprocessing'].getint('n_fft')
        self.precision = config['preprocessing'].getint('precision')
        self.delta_delta = config['preprocessing'].getboolean('delta_delta')
        self.rate = config['preprocessing'].getint('rate')
        self.step_length = config['preprocessing'].getfloat('step_size')                        # Given in seconds
        self.sample_length_scalogram = config['preprocessing'].getfloat('step_size_scalogram')    # Given in percentages
        self.sample_length = int(self.rate * self.step_length)                                  # Given in samples


        self.activate_envelope = config['preprocessing'].getboolean('activate_thresholding')
        self.threshold = config['preprocessing'].getfloat('threshold')
        self.audio_folder = config['preprocessing']['audio_folder']
        self.file_type = os.listdir(f'../Datasets/audio/{self.audio_folder}/fold1/')[0].split('.')[-1]

        if self.file_type == 'wav':
            assert self.step_length <= config['preprocessing'].getfloat('signal_minimum_length'), \
                f'step_length is larger than signal_length. ' \
                f'Please set a value for step_length that is lower than signal_length'

        # Find the number of possible samples from each of the folds for training, validation and testing
        self.no_test_fold = config['model'].getboolean('no_test_fold')
        if self.no_test_fold is False:
            self.n_training_samples = self.find_samples_per_epoch(self.df, start_fold=1, end_fold=8)
            self.n_validation_samples = self.find_samples_per_epoch(self.df_v, start_fold=9, end_fold=9)
            self.n_testing_samples = self.find_samples_per_epoch(self.df_v, start_fold=10, end_fold=10)
        else:
            self.n_training_samples = self.find_samples_per_epoch(self.df, start_fold=1, end_fold=9)
            self.n_validation_samples = self.find_samples_per_epoch(self.df_v, start_fold=10, end_fold=10)

        self.training_seed, self.validation_seed = np.random.randint(0, 100000, 2)
        self.testing_seed = 42
        self.scales = np.arange(1, 151)
        # Use the sample numbers to create probability distribution for the labels
        self.class_dist = [self.n_training_samples[classes] / sum(self.n_training_samples.values())
                           for classes in self.classes]
        self.prob_dist = pd.Series(self.class_dist, self.classes)

        self.chosen_class_dist = {}
        self.validation_fold = None

    def filter_dataframe(self, df, df_v):
        """
        Filters out the dataframe based on the specifications from the config file
        :return:
        """

        # Filter out short audio
        df = df.loc[df['length'] >= config['preprocessing'].getfloat('signal_minimum_length')]
        df_v = df_v.loc[df_v['length'] >= config['preprocessing'].getfloat('signal_minimum_length')]

        # Filter out unused params
        total_of_params = self.time_shift_param
        total_of_params.extend(self.pitch_shift_param)
        total_of_params.append(float('NaN'))
        try:
            df = df[df.param.isin(total_of_params)]
        except AttributeError:
            return df, df_v

        # Filter out augmentation
        if 'time_shift' not in self.augmentations:
            df = df[df.augmentation != 'time']
        if 'pitch_shift' not in self.augmentations:
            df = df[df.augmentation != 'pitch']

        # Reset the indexes
        df = df.reset_index()
        df_v = df_v.reset_index()

        return df, df_v

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

    def find_samples_per_epoch(self, df, start_fold=1, end_fold=9):
        """
        Finds all the files that exist in the data set for a particular class.
        It then checks the defined step length and checks how many possible samples that
        that can be extracted for the class.

        Example:    If the audio file is 4s and the step length is set to 0.5s then the number of possible samples
                    is 8. If a class has 16 audio files of 4s then the number of possible samples is a total of
                    4s/0.5s * 16 files = 128 samples

        :param df: dataframe to use
        :param start_fold: Start fold to start checking from
        :param end_fold: The last fold to check. Function checks all folds in between
        :return:
        """
        samples_dict = {}

        # Loop through folds
        for f in range(start_fold, end_fold+1):

            # Find all the files in the fold
            files_in_fold = df[df.fold == f]

            for label in self.classes:
                # Find all the matches of that class in the fold
                files_with_class = files_in_fold[df.label == label]

                if self.file_type == 'jpeg':
                    if label in samples_dict.keys():
                        samples_dict[label] += len(files_with_class)
                    else:
                        samples_dict[label] = len(files_with_class)
                else:
                    # Add up the possible samples
                    if label in samples_dict.keys():
                        samples_dict[label] += int(files_with_class['length'].sum() / self.step_length)
                    else:
                        samples_dict[label] = int(files_with_class['length'].sum() / self.step_length)

        return samples_dict

    def extract_feature(self, file_path, feature_to_extract='mfcc',
                        activate_threshold=False, seed=None, delta_delta=False, random_extraction=True):
        """
        Reads the signal from the file path for then to extract one of several spectral features.
        Available options are: mfcc, msfb, spectogram, scalogram and fbank. Can either be extracted from a wav
        file. Has the option to simply read image files of spectograms if they ahve been extracted in advance. This is
        potentially much faster.
        Available preprocessing options: enveloping, random_extraction and delta-delta
        :param random_extraction: Activate to extract a random clip of the sample
        :param delta_delta: Append the delta_delta to the end of the feature vectors
        :param file_path:
        :param seed: None by default
        :param feature_to_extract: Choose what feature to extract. Current alternatives: 'mffc', 'fbank', scalogram and
         'msfb'
        :param file_path: File path to audio file
        :param activate_threshold: A lower boundary to filter out weak signals
        :return:
        """
        # If file is wav file
        if self.file_type == 'wav':
            sample, rate = sf.read(file_path)
            if random_extraction is True:
                # Choose a window of the signal to use for the sample
                try:
                    if feature_to_extract != 'scalogram':
                        sample = self.select_random_audio_clip(sample, seed)

                except ValueError:
                    print("audio file to small. Skip and move to next")
                    return 'move to next file'

            # Perform filtering with a threshold on the time signal
            if activate_threshold is True:
                mask = self.envelope(sample, rate, self.threshold)
                sample = sample[mask]

        # If file is jpeg file
        elif self.file_type == 'jpeg':
            with Image.open(file_path + '.jpeg') as sample:
                # Convert image to numpy array and resize so all are the same spatial size.
                sample_hat = np.asarray(sample)
            if random_extraction is True:
                # Choose a window of the signal to use for the sample
                np.random.seed(seed)
                rand_index = np.random.randint(0, sample_hat.shape[0] - sample_hat.shape[0]*self.sample_length_scalogram)
                return sample_hat[:, rand_index:rand_index + self.sample_length]

        # Extracts the mel frequency cepstrum coefficients
        if feature_to_extract == 'mfcc':
            if self.file_type == 'wav':
                sample_hat = mfcc(sample, rate,
                              numcep=self.n_feat,
                              nfilt=self.n_filt,
                              nfft=self.n_fft).T
            elif self.file_type == 'jpeg':
                width = 399
                height = 13
                dim = (width, height)
                sample_hat = cv2.resize(sample_hat, dim, interpolation=cv2.INTER_AREA)

        # Extract the log mel frequency filter banks
        elif feature_to_extract == 'msfb':
            if self.file_type == 'wav':
                sample_hat = logfbank(sample, rate,
                                      nfilt=self.n_filt,
                                      nfft=self.n_fft).T
            elif self.file_type == 'jpeg':
                width = 399
                height = 26
                dim = (width, height)
                sample_hat = cv2.resize(sample_hat, dim, interpolation=cv2.INTER_AREA)

        # Extract the mel frequency filter banks
        elif feature_to_extract == 'fbank':
            if self.file_type == 'wav':
                sample_hat = fbank(sample, rate,
                                   nfilt=self.n_filt,
                                   nfft=self.n_fft)[0].T
            elif self.file_type == 'jpeg':
                width = 398
                height = 601
                dim = (width, height)
                sample_hat = cv2.resize(sample_hat, dim, interpolation=cv2.INTER_AREA)

        # Extract the log of the power spectrum
        elif feature_to_extract == 'spectogram':
            if self.file_type == 'wav':
                sample = sigproc.preemphasis(sample, 0.97)
                _, _, Sxx = spectrogram(sample, rate, noverlap=240,
                                               nfft=self.n_fft,
                                               window=get_window('hamming', 400, self.n_fft))
                sample_hat = np.where(Sxx == 0, np.finfo(float).eps, Sxx)
                sample_hat = np.log(sample_hat)
            elif self.file_type == 'jpeg':
                width = 398
                height = 601
                dim = (width, height)
                sample_hat = cv2.resize(sample_hat, dim, interpolation=cv2.INTER_AREA)


        # Extract the wavelet transform of the signal at different scales
        elif feature_to_extract == 'scalogram':
            if self.file_type == 'wav':
                sample_hat = cwt(sample_hat, 'morlet')
            elif self.file_type == 'jpeg':
                # Due to a scaling error, 4 scales needs to be dropped in height
                width = 500
                height = 141
                dim = (width, height)
                sample_hat = cv2.resize(sample_hat, dim, interpolation=cv2.INTER_AREA)


        else:
            if self.file_type == 'wav':
                raise ValueError('Please choose an existing feature in the config.ini file:'
                                 '\n    - MFCC'
                                 '\n    - msfb, '
                                 '\n    - Spectogram '
                                 '\n    - Fbank'
                                 '\n    - Scalogram')

        # Apply the change of the dynamics to the feature vector
        if delta_delta is True:
            d = delta(sample_hat.T, 2)
            sample_hat = np.append(sample_hat.T, d, axis=0)

        return sample_hat

    def generate_labels(self, seed=42):
        """
        Used with the generator to create the labels for the predict_generator during testing.
        This is to ensure that the same labels are chosen every time.
        :param seed:
        :return:
        """
        # files = []
        data = {'files': [], 'labels': []}
        np.random.seed(seed)

        # Find the files in the current fold
        folds = np.unique(self.df_v['fold'])
        folds = np.roll(folds, folds[-1] - self.fold)
        testing_folds = folds[-1]
        files_in_fold = self.df_v.loc[self.df_v.fold == testing_folds]

        # Loop through the files in the fold and create a batch
        for n in range(sum(self.n_testing_samples.values())):

            # Pick a random class from the probability distribution and then a random file with that class
            rand_class = np.random.choice(self.classes, p=self.prob_dist)

            # Pick a random file with that class
            file = np.random.choice(files_in_fold[self.df_v.label == rand_class].slice_file_name)
            file_path = f'../Datasets/audio/{self.audio_folder}/fold{testing_folds}/{file}'

            # Append to dict
            data['files'].append(file_path)
            data['labels'].append(rand_class)

        # One-hot encode the labels and append
        data['labels'] = np.array(data['labels'])
        # labels = to_categorical(labels, num_classes=len(self.classes))
        return data

    def generate_data_for_predict_generator(self, labels_to_predict):
        """
        This is the generator to use for the predict generator. This is used when testing models. It does not
        produce labels on the spot, but  rather takes in a list of labels to generate batches for.
        :param labels_to_predict:
        :return:
        """
        # Initialize min and max for x
        _min, _max = float('inf'), -float('inf')

        # Batch counter
        n = 0

        # While loop that generate batches of data
        while True:
            x = []  # Set up lists

            # Loop through the files in the fold and create a batch
            for i in range(self.batch_size_for_test):
                file_path = labels_to_predict['files'][n+i]

                # Extract feature from signal
                x_sample = self.extract_feature(file_path,
                                                feature_to_extract=self.feature,
                                                activate_threshold=self.activate_envelope,
                                                seed=None,
                                                delta_delta=self.delta_delta,
                                                random_extraction=self.random_extraction)

                # Update min and max values
                _min = min(np.amin(x_sample), _min)
                _max = max(np.amax(x_sample), _max)

                # Create batch set with corresponding labels
                x.append(x_sample)

            # Normalize X and reshape the features
            x = np.asarray(x)
            x = (x - _min) / (_max - _min)  # Normalize x and y
            if self.file_type == 'jpeg':
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)  # Reshape all to same size - scalogram
            elif self.file_type == 'wav':
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)  # Reshape all to same size - other features

            n += self.batch_size_for_test
            yield x

    def preprocess_dataset_generator(self, mode='training'):
        """
        Pre-processes a batch of data which is yielded for every function call. Size is defined by batch size
        is defined the config.ini file. If mode = 'training', the first call will make a batch from the first
        available fold of the training folds. Next call will be from the next available fold. When the end
        is reached it will start from the first fold again.
        :return:
        """

        _min, _max = float('inf'), -float('inf')    # Initialize min and max for x
        folds = np.unique(self.df['fold'])          # The folds to loop through
        i = 0
        # Separate folds into training, validation and split
        folds = np.roll(folds, folds[-1] - self.fold)
        if self.no_test_fold is False:
            training_folds = folds[:-2]
            validation_folds = folds[-2]
        else:
            training_folds = folds[:-1]
            validation_folds = folds[-1]

        self.validation_fold = self.fold

        # Choose fold to start from
        if mode == 'training':
            fold = training_folds
            np.random.seed(self.training_seed)
        elif mode == 'validation':
            fold = validation_folds
            np.random.seed(self.validation_seed)

        # Build feature samples
        while True:
            x, y = [], []  # Set up lists

            # Find the files in the current fold
            if mode == 'training':
                files_in_fold = self.df.loc[self.df.fold == fold[i]]
            elif mode == 'validation':
                files_in_fold = self.df_v.loc[self.df_v.fold == validation_folds]

            # Counter that checks if batch_size is reached
            counter = 0
            # Loop through the files in the fold and create a batch
            for n in range(self.batch_size):

                # Pick a random class from the probability distribution and then a random file with that class
                rand_class = np.random.choice(self.classes, p=self.prob_dist)
                file = np.random.choice(files_in_fold[self.df.label == rand_class].slice_file_name)

                if mode == 'training':
                    file_path = f'../Datasets/audio/{self.audio_folder}/fold{fold[i]}/{file}'
                elif mode == 'validation':
                    file_path = f'../Datasets/audio/{self.audio_folder}/fold{validation_folds}/{file}'

                # Extract feature from signal
                x_sample = self.extract_feature(file_path,
                                                feature_to_extract=self.feature,
                                                activate_threshold=self.activate_envelope,
                                                delta_delta=self.delta_delta,
                                                random_extraction=self.random_extraction
                                                )

                # If the flag is set, it means it could not process current file and it moves to next.
                if x_sample == 'move to next file':
                    print(rand_class)
                    continue

                # Update min and max values
                _min = min(np.amin(x_sample), _min)
                _max = max(np.amax(x_sample), _max)

                # Create batch set with corresponding labels'
                x.append(x_sample)
                y.append(self.classes.index(rand_class))

                counter += 1
                # Check if batch size is reached
                if counter >= self.batch_size:
                    break

            # Normalize X and y and reshape the features
            y = np.asarray(y)
            x = np.array(x)
            x = (x - _min) / (_max - _min)                              # Normalize x and y
            if self.file_type == 'jpeg':
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)    # Reshape all to same size - scalogram
            elif self.file_type == 'wav':
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Reshape all to same size - other features

            # One-hot encode the labels and append
            y = to_categorical(y, num_classes=len(self.classes))

            # Reset fold number when reaching the end of the training folds
            if mode == 'training':
                i += 1
                if i >= len(training_folds):
                    i = 0

            # Return data and labels
            yield x, y

    @staticmethod
    def envelope(y, rate, threshold):
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

    @staticmethod
    def make_signal_mono(y):
        """
        If a signal is stereo, average out the channels and make the signal mono
        :return:
        """
        if len(y.shape) > 1:
            y = y.reshape((-1, y.shape[1])).T
            y = np.mean(y, axis=0)
        return y

    @staticmethod
    def resample_signal(y, orig_sr, target_sr):
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
        return np.ascontiguousarray(y_hat)

    def downsample_all_signals(self, df, target_sr=8000):
        """
        Loops through all the sound files and applies a high pass filter
        :return:
        """
        params = {'time_shift': self.time_shift_param, 'pitch_shift': self.pitch_shift_param}
        import time
        # Store in clean directory

        for wav_file in tqdm(df.slice_file_name):
            # Find filename and filepath
            fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
            file_name = f'../Datasets/audio/downsampled/fold{fold}/{wav_file}'

            # Read file, monotize if stereo and resample
            signal, sr = sf.read(file_name)
            signal_hat = preprocessing.resample_signal(signal, orig_sr=sr, target_sr=target_sr)

            # print(len(signal_hat)/8000)
            # for deformation in self.augmentations:
                # for param in params[deformation]:
                #     signal_hat = self.deform_signal(signal, target_sr, deformation, param)
                #
                #     # Add the new file to the CSV file
                #     row_info = df.loc[df['slice_file_name'] == wav_file]
                #     org_name = re.findall('(.+).wav', wav_file)[0]
                #     row_info['slice_file_name'] = f"{org_name}_{deformation}_{param}.wav"
                #     row_info['length'] = len(signal_hat)/target_sr
                #
                #     df = df.append(row_info, ignore_index=True)

            # Write to file
            wavfile.write(filename=f'../Datasets/audio/8Hz/fold{fold}/{wav_file}',
                          rate=target_sr,
                          data=signal_hat)

            # df.to_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length_augmented.csv')

    @staticmethod
    def add_noise(data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def stretch_signal(self, data, rate=0.8):

        input_length = len(data)
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
            # Remove the padding
            mask = [self.envelope(data, 16000, 0.00001)]
            data = data[mask]

        return data

    def deform_signal(self, y, sr, deformation, param):
        """
        Apply augmentation to a signal.
        Possible choices so far are "time_shift" and "pitch_shift".
        Parameters are adjusted in the config file under [augmentation].
        :param y: data
        :param sr: sample rate
        :return: dict[time_shifts, pitch_shifts], dict([], []) if no params is set or both are set to False
        """

        # deformation = np.random.choice(self.augmentations)

        if deformation == 'time_shift':
            param = np.random.choice(self.time_shift_param)
            y = pyrb.time_stretch(y, sr, param)
            # self.data_aug['downsampled'] += 1

        elif deformation == 'pitch_shift':
            param = np.random.choice(self.pitch_shift_param)
            y = pyrb.pitch_shift(y, sr, param)
            # self.data_aug['downsampled'] += 1

        elif deformation == 'both':
            param1 = np.random.choice(self.pitch_shift_param)
            param2 = np.random.choice(self.time_shift_param)
            y = pyrb.pitch_shift(y, sr, param1)
            y = pyrb.time_stretch(y, sr, param2)
            # self.data_aug['downsampled'] += 1

        # elif deformation == 'None':
        #     self.data_aug['normal'] += 1

        return y


if __name__ == '__main__':
    """
    Main function
    :return:
    """




    # import scipy.signal as sig
    # ax = np.arange(0, 1000)
    # from math import exp
    # voiceperocct = 10
    # octaves = 5
    # a0 = 2**(1/voiceperocct)
    # fs = 16000
    # scales = []
    # len_of_scales = octaves*voiceperocct-voiceperocct

        # a0 = a0**(octaves*voiceperocct)
        # scales.append(a0)

    # ax = np.arange(1, 151)

    # scales = []
    #
    # scales2 = []
    # for f, x in zip(freqs, ax):
    #     scales.append(pywt.central_frequency('morl', 8)/f)
    #     scales2.append(pywt.scale2frequency('morl', x, 8))

    # Create dataframe
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_augmented.csv')

    # Create a class distribution
    class_dist = df.groupby(['label'])['length'].mean()

    preprocessing = PreprocessData(df)
    # Fetch the classes from the CSV file
    classes = list(np.unique(df.label))
    # save_scalograms(preprocessing, df)
    # Dicts for things we want to store
    signals = {}
    mfccs = {}
    filterbank = {}
    f_bank = {}
    fft = {}
    mfccs_low = {}
    filterbank_low = {}
    mfccs_low_no_mask = {}
    msfb_low_no_mask = {}
    Sxx = {}

    c = 'gun_shot'

    classes = ['siren',
               'gun_shot',
               'jackhammer',
               'street_music',
               'children_playing',
               'drilling',
               'car_horn',
               'air_conditioner',
               'engine_idling',
               'dog_bark']
    plotting_functions.plot_performance(r'C:\Users\toanb\OneDrive\skole\UiO\Master\code\experiments_results\exp2_three_diff_class\spectogram\run-.-tag-val_acc.csv', smoothing=0)
    exit()

    for c in classes:
        # Get the file name and the fold it exists in from the dataframe
        wav_file = df[df.label == c].iloc[0, 0]
        fold = df.loc[df['slice_file_name'] == wav_file, 'fold'].iloc[0]
        length = df.loc[df['slice_file_name'] == wav_file, 'length'].iloc[0]
        target_sr = 16000
        file_path = f'../Datasets/audio/{preprocessing.audio_folder}/fold{fold}/{wav_file}'
        # Read signal and add it to dict.
        signal, sr = sf.read(f'../Datasets/audio/augmented/fold{fold}/{wav_file}')

        feature = preprocessing.extract_feature(file_path, 'spectogram', random_extraction=False)




        # signal, sr = sf.read('C:/Users/toanb/OneDrive/skole/UiO/Master/Datasets/audio/augmented/fold1/7383-3-0-0.wav')
        file = r'C:\Users\toanb\OneDrive\skole\UiO\Master\Datasets\Audio-Classification-master\wavfiles\0c67f402.wav'
        jam = jams.JAMS()
        jam.file_metadata.duration = length
        # Make empty jams
        j_orig = muda.load_jam_audio(jams.JAMS(), file)

        file_bg = r'C:\Users\toanb\OneDrive\skole\UiO\Master\Datasets\urbansounds_jams\UrbanSound8K-JAMS\background_noise\150993__saphe__street-scene-1.wav'

        background = muda.deformers.BackgroundNoise(1, file_bg)
        jam_out = background.transform(j_orig)
        # jam_out.sandbox.muda._audio
        # sf.write("bikkje.wav", jam_out, samplerate=16000)
        muda.save("test.wav", 'test.jam', jam_out)

        exit()

        # feature_scalo = preprocessing.extract_feature(signal, sr, 'scalogram', random_extraction=False)

    exit()








