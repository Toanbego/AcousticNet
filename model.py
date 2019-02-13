import os
import argparse
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc, fbank, logfbank
import python_speech_features
import soundfile as sf
import preprocess_data



def parse_arguments():
    """
    Choose what model to use. Either 'conv' for convolutional model
    or 'time' for a recurrent lstm model.
    :return:
    """
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('--mode', '-m', type=str,
                        help='Either conv or time',
                        default='conv')

    args = parser.parse_args()
    return args


def build_feature_from_signal(file_path, config, feature_to_extract='mfcc'):
    """
    Reads the signal from the file path. Then build mffc features that is returned.
    If the signal is stereo, the signal will be split up and only the first channel is used.

    Later implementations should consist of using both channels, and being able to select other features than mfccs
    :param file_path:
    :return:
    """
    # Read file
    wav, rate = sf.read(file_path)

    # If len is > 1 it means the signal is stereo and needs to be split up
    if len(wav.shape) > 1:
        wav, signal_2 = preprocess_data.separate_stereo_signal(wav)

    # Start from a random point in the signal and build a mffc feature
    rand_index = np.random.randint(0, wav.shape[0] - config.step)
    sample = wav[rand_index:rand_index + config.step]

    # Extracts the mel frequency cepstrum coefficients
    if feature_to_extract == 'mfcc':
        sample = mfcc(sample, rate,
                        numcep=config.nfeat,
                        nfilt=config.nfilt,
                        nfft=config.nfft).T

    elif feature_to_extract == 'logfbank':
        sample = logfbank(sample, rate,
                        # numcep=config.nfeat,
                        nfilt=config.nfilt,
                        nfft=config.nfft).T

    return sample


def build_features_for_training(config, df, n_samples, classes, class_dist, prob_dist):
    """
    Builds and shapes the data according to the mode chosen.

    Randomly samples the audio sample every n/second.
    The sample is converted to a spectogram and appended into an list of X and Y
    and is to be used as features to train the model.
    A normal sample length could be 1/10th of a second, but one could
    always try more or less.
    :return:
    """
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    df.set_index('slice_file_name', inplace=True)
    feature = 'mfcc'
    print(f"Features used are {feature}")
    # Build feature samples
    for _ in tqdm(range(n_samples)):

        # Pick a random class from the probability distribution and then a random file with that class
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label == rand_class].index)

        # Find the fold and the file
        fold = df.loc[df.index == file, 'fold'].iloc[0]
        file_path = f'../Datasets/audio/downsampled/fold{fold}/{file}'

        # Get the mffc feature
        x_sample = build_feature_from_signal(file_path, config, feature_to_extract=feature)

        # Update min and max
        _min = min(np.amin(x_sample), _min)
        _max = max(np.amax(x_sample), _max)
        X.append(x_sample if config.mode == 'conv' else x_sample.T)
        y.append(classes.index(rand_class))

    # Normalize X and y
    y = np.array(y)
    X = np.array(X)
    X = (X - _min) / (_max - _min)

    # Reshape X based on 'conv' mode or 'time' mode
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    y = to_categorical(y, num_classes=10)
    return X, y


def get_conv_model(input_shape):
    """
    Create a convolutional model for deep learning
    :return:
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same',))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', ))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same', ))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model


def get_recurrent_model(input_shape):
    """
    The Recurrent Network model
    Layers:
        - Two LSTM layers
        - Dropout of 0.5
        - Four time distributed layers
        - A flatten
        - Dense output layer with softmax
    Compiles with:
        - Categorical crossentropy
        - Adam
        - Accuracy metric
    :return:
    """
    # Shape of data for RNN is (n, time, features)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model


class Config:
    """
    Class for hyper parameters
    """
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=1103, rate=16000):

        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat  # Same as number of cepstrums for mffc
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)


def main():

    args = parse_arguments()
    # Create dataframe
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')

    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    # Is set to 1000 for testing purposes
    # n_samples = 2 * int(df['length'].sum()/0.1)
    n_samples = 300

    prob_dist = class_dist / class_dist.sum()

    # Set up the config class
    config = Config(mode=args.mode)

    if config.mode == 'conv':
        # Goes through the folder with the data and randomly extracts data samples from the audio files
        x, y = build_features_for_training(config, df, n_samples, classes, class_dist, prob_dist)
        # Reshape one-hot encode matrix back to string labels
        y_flat = np.argmax(y, axis=1)
        input_shape = (x.shape[1], x.shape[2], 1)
        model = get_conv_model(input_shape)

    elif config.mode == 'time':
        x, y = build_features_for_training(config, df, n_samples, classes, class_dist, prob_dist)
        y_flat = np.argmax(y, axis=1)
        input_shape = (x.shape[1], x.shape[2])
        model = get_recurrent_model(input_shape)

    class_weight = compute_class_weight('balanced',
                                        np.unique(y_flat),
                                        y_flat)
    # Train the network
    model.fit(x, y, epochs=15,
              batch_size=32,
              shuffle=True,
              class_weight=class_weight)


main()






