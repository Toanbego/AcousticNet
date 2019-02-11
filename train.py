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



class audio:
    """
    Class object for training a model to classify acoustic sounds
    """
    def __init__(self, df):
        """
        Initialize variables
        """

        # Find classes and create class distribution
        self.classes = list(np.unique(df['label']))
        self.class_dist = df.groupby(['label'])['length'].mean()

        # Audio parameters
        self.mode = 'conv'
        self.nfilt = 26
        self.nfeat = 13
        self.nfft = 1200
        self.rate = 22050
        self.step = int(self.rate / 5)

    def convolutional_model(self, input_shape):
        """
        A novel convolutional model network
        :return:
        """

        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                         padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                         padding='same', ))
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

    def build_features(self):
        """
        Generator that
        :return:
        """

    def split_data(self):
        """
        Split the folds into training, validation and test
        :return:
        """

    def train_model(self, epochs=20):
        """
        Trains a model
        :return:
        """




def main():
    """
    Steps:
        1. Read csv and load data
        2. Split data into train, test, validation.
           It is important to not train on the validation and test data
           so consider using one fold for test and one for validation. Should
           work good for a k-fold algorithm
        3. Extract features. Loop over audio files with a certain
           window length and randomly extract spectograms to create the X vector.
           Train on this data. Consider extracting features once and save them so this
           task don't have to be repeated over and over
        4. Create a convolutional model. A simple one will do for beginners. At this point,
           make sure it trains and improves accuracy from the data.
        5. Implement validation and testing algorithms
        6. When this pipeline is finished, work with the experimentation!

    """
    # Read csv for urbanSounds
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')

    audio_model = audio(df)










main()
