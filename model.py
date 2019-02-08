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
from python_speech_features import mfcc
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


def build_rand_feat(config, df, n_samples, classes, class_dist, prob_dist):
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
    i = 0

    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label == rand_class].index)
        fold = df.loc[df.index == file, 'fold'].iloc[0]
        wav, rate = sf.read(f'../Datasets/audio/fold{fold}/{file}')
        # rate, wav = sf.read('clean/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate,
                        numcep=config.nfeat,
                        nfilt=config.nfilt,
                        nfft=config.nfft).T

        # Update min and max
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))


    # Normalize X and y
    for i, array in enumerate(X):
        if array.shape != np.zeros((13, 9)).shape:
            del X[i]
        else:
            print(array)

    for i, array in enumerate(X):
        if array.shape != np.zeros((13, 9)).shape:
            print(array.shape)
            del X[i]

    for i, array in enumerate(X):
        if array.shape != np.zeros((13, 9)).shape:
            print(array.shape)
            del X[i]

    for i, array in enumerate(X):
        if array.shape != np.zeros((13, 9)).shape:
            print(array.shape)
            del X[i]

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
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=2400, rate=22050):

        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)


def main():

    args = parse_arguments()
    # Create dataframe
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K.csv')

    # Format data and add a 'length' column
    df.set_index('slice_file_name', inplace=True)
    df = preprocess_data.add_length_to_column(df)

    classes = list(np.unique(df.label))
    class_dist = df.groupby(['label'])['length'].mean()

    # Is set to 1000 for testing purposes
    # n_samples = 2 * int(df['length'].sum()/0.1)
    n_samples = 200
    prob_dist = class_dist / class_dist.sum()

    list_of_arrays = []

    for i in range(1000):
        add_diff_shape = np.random.randint(0, 100)
        if add_diff_shape > 90:
            new_shape = np.random.randint(6, 14)
            list_of_arrays.append(np.random.rand(13, new_shape))
        else:
            list_of_arrays.append(np.random.rand(13, 9))


    for i, array in enumerate(list_of_arrays):
        if array.shape != np.zeros((13, 9)).shape:
            del list_of_arrays[i]

        else:
            print(array.shape)

    for i, array in enumerate(list_of_arrays):
        if array.shape != np.zeros((13, 9)).shape:
            print(array.shape)
            del list_of_arrays[i]
    list_of_arrays = np.array(list_of_arrays)


    choices = np.random.choice(class_dist.index, p=prob_dist)

    # Plot the label distribution
    fig, ax = plt.subplots()
    ax.set_title('Class Distribution', y=1.08)
    ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
           shadow=False, startangle=90)
    ax.axis('equal')
    plt.show()

    # Set up the config class
    config = Config(mode=args.mode)

    if config.mode == 'conv':
        X, y = build_rand_feat(config, df, n_samples, classes, class_dist, prob_dist)

        # Reshape one-hot encode matrix back to string labels
        y_flat = np.argmax(y, axis=1)
        input_shape = (X.shape[1], X.shape[2], 1)
        model = get_conv_model(input_shape)

    elif config.mode == 'time':
        X, y = build_rand_feat(config, df, n_samples, classes, class_dist, prob_dist)
        y_flat = np.argmax(y, axis=1)
        input_shape = (X.shape[1], X.shape[2])
        model = get_recurrent_model(input_shape)

    class_weight = compute_class_weight('balanced',
                                        np.unique(y_flat),
                                        y_flat)
    # Train the network
    model.fit(X, y, epochs=15,
              batch_size=32,
              shuffle=True,
              class_weight=class_weight)


main()






