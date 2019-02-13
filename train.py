import configparser
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.callbacks as callbacks
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc, fbank, logfbank
import soundfile as sf
import preprocess_data

# Parse the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")


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


class ClassifyAudio:
    """
    Class object for training a model to classify acoustic sounds
    """
    def __init__(self, df):
        """
        Initialize variables
        """

        config['simulation'].getboolean('train')
        # DataFrame
        self.df = df

        # Find classes and create class distribution
        self.classes = list(np.unique(df['label']))
        self.class_dist = df.groupby(['label'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()

        # Audio parameters
        self.mode = config['model']['net']
        self.save_features = config['preprocessing']['save_features']
        self.load_features = config['preprocessing']['load_features']
        self.feature = config['preprocessing']['feature']
        self.n_filt = config['preprocessing'].getint('n_filt')
        self.n_feat = config['preprocessing'].getint('n_feat')
        self.n_fft = config['preprocessing'].getint('n_fft')
        self.rate = config['preprocessing'].getint('rate')
        self.step = int(self.rate / 10)

        # Training parameters
        self.n_samples = config['model'].getint('n_samples')
        self.epochs = config['model'].getint('epochs')
        self.optimizer = config['model']['optimizer']

        # Parameters to be initiated at a later stage
        self.class_weight = None
        self.input_shape = None
        self.model = None

    def set_up_model(self, train_x):
        """
        Methods compiles the specified model. Currently only CNN is available.
        :return:
        """
        # Define input shape and compile model
        self.input_shape = (train_x.shape[1], train_x.shape[2], 1)
        self.model = self.convolutional_model()

    def convolutional_model(self):
        """
        A novel convolutional model network
        :return:
        """

        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1),
                         padding='same', input_shape=self.input_shape))
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
                      optimizer='self.optimizer',
                      metrics=['acc'])

        return model

    def build_feature_from_signal(self, file_path, feature_to_extract='mfcc'):
        """
        Reads the signal from the file path. Then build mffc features that is returned.
        If the signal is stereo, the signal will be split up and only the first channel is used.

        Later implementations should consist of using both channels, and being able to select other features than mfccs
        :param feature_to_extract: Choose what feature to extract. Current alternatives: 'mffc', 'fbank' and 'logfbank'
        :param file_path:
        :return:
        """
        # Read file
        sample, rate = sf.read(file_path)
        # TODO: Find the minimum length of a signal as use that as step length
        #       Right now the smallest signals of 0.05 seconds is just skipped. Though there are only like three.

        # If len is > 1 it means the signal is stereo and needs to be split up
        if len(sample.shape) > 1:
            sample, signal_2 = preprocess_data.separate_stereo_signal(sample)

        try:
            rand_index = np.random.randint(0, sample.shape[0] - self.step)
            sample = sample[rand_index:rand_index + self.step]
        except ValueError:
            print("audio file to small. Skip and move to next")
            return 'move to next file'

        # Extracts the mel frequency cepstrum coefficients
        if feature_to_extract == 'mfcc':
            sample = mfcc(sample, rate,
                          numcep=self.n_feat,
                          nfilt=self.n_filt,
                          nfft=self.n_fft).T

        # Extract the log mel frequency filter banks
        elif feature_to_extract == 'logfbank':
            sample = logfbank(sample, rate,
                              nfilt=self.n_filt,
                              nfft=self.n_fft).T

        # Extract the mel frequency filter banks
        elif feature_to_extract == 'fbank':
            sample = fbank(sample, rate,
                           nfilt=self.n_filt,
                           nfft=self.n_fft).T
        else:
            print('Choose an existing feature: mfcc, logfbank or fbank')
            raise ValueError
        return sample

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
        x = []
        y = []
        _min, _max = float('inf'), -float('inf')
        # self.df.set_index('slice_file_name', inplace=True)

        print(f"Features used are {self.feature}")

        # Get a list of each fold
        folds = list(np.unique(self.df['fold']))

        # Build feature samples
        for fold in folds:
            print(f'\nExtracting data from fold{fold}')
            # Get the filenames that exists in that fold
            files_in_fold = self.df.loc[self.df.fold == fold].slice_file_name

            # Loop through the files in the fold
            for file in tqdm(files_in_fold):
                # TODO: Create n_samples from fold. Create them by randomly choosing
                #       index and going step_length from there.

                # Get the label from the column by using the file name
                label = self.df.loc[self.df['slice_file_name'] == file, 'label'].iloc[0]

                # Set file path
                file_path = f'../Datasets/audio/downsampled/fold{fold}/{file}'

                # Get the mffc feature
                x_sample = self.build_feature_from_signal(file_path, feature_to_extract=self.feature)
                if x_sample == 'move to next file':
                    continue
                # Update min and max
                _min = min(np.amin(x_sample), _min)
                _max = max(np.amax(x_sample), _max)
                x.append(x_sample if self.mode == 'conv' else x_sample.T)
                y.append(self.classes.index(label))

        # Normalize X and y
        y = np.array(y)
        x = np.array(x)
        x = (x - _min) / (_max - _min)

        # Reshape X based on 'conv' mode or 'time' mode
        if self.mode == 'cnn':
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        # One-hot encode the labels
        y = to_categorical(y, num_classes=10)

        # Computes the class weight for the samples extracted
        y_flat = np.argmax(y, axis=1)  # Reshape one-hot encode matrix back to string labels
        self.class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

        return x, y

    def create_training_data_shuffle(self):
        """
        Builds and shapes the data according to the mode chosen.

        UrbanSounds homepage encourages not to shuffle the data, and mix the folds together.
        The sample is converted to a spectrogram and appended into an list of X and Y
        and is to be used as features to train the model.
        A normal sample length could be 1/10th of a second, but one could
        always try more or less.
        :return:
        """
        x = []
        y = []
        _min, _max = float('inf'), -float('inf')
        self.df.set_index('slice_file_name', inplace=True)

        feature = 'mfcc'
        print(f"Features used are {feature}")
        # Build feature samples
        for _ in tqdm(range(self.n_samples)):

            # Pick a random class from the probability distribution and then a random file with that class
            rand_class = np.random.choice(self.class_dist.index, p=self.prob_dist)
            file = np.random.choice(self.df[self.df.label == rand_class].index)

            # Find the fold and the file
            fold = self.df.loc[self.df.index == file, 'fold'].iloc[0]
            file_path = f'../Datasets/audio/downsampled/fold{fold}/{file}'

            # Get the mffc feature
            x_sample = self.build_feature_from_signal(file_path, feature_to_extract=feature)

            # Update min and max
            _min = min(np.amin(x_sample), _min)
            _max = max(np.amax(x_sample), _max)
            x.append(x_sample if self.mode == 'conv' else x_sample.T)
            y.append(self.classes.index(rand_class))

        # Normalize X and y
        y = np.array(y)
        x = np.array(x)
        x = (x - _min) / (_max - _min)

        # Reshape X based on 'conv' mode or 'time' mode
        if self.mode == 'cnn':
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        # One-hot encode the labels
        y = to_categorical(y, num_classes=10)

        # Computes the class weight for the samples extracted
        y_flat = np.argmax(y, axis=1)  # Reshape one-hot encode matrix back to string labels
        self.class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

        if self.save_features:
            np.savez('')


        return x, y

    def train(self, train_x, train_y):
        """
        Method used for training a model.
        :return:
        """
        # Train the network
        # self.model.fit(train_x[:-838], train_y[:-838],
        #                epochs=self.epochs,
        #                batch_size=32,
        #                shuffle=False,
        #                class_weight=self.class_weight,
        #                validation_data=(train_x[-838:], train_y[-838:]))

        self.model.fit(train_x, train_y,
                       epochs=self.epochs,
                       batch_size=32,
                       shuffle=True,
                       class_weight=self.class_weight,
                       # validation_data=(train_x, train_y)
                       )

def main():
    """
    Steps:
        1. Read csv and load data
        2. Split data into train, test, validation.
           It is important to not train on the validation and test data
           so consider using one fold for test and one for validation. Should
           work good for a k-fold algorithm
        3. Extract features. Loop over audio files with a certain
           window length and randomly extract spectrograms to create the X vector.
           Train on this data. Consider extracting features once and save them so this
           task don't have to be repeated over and over
        4. Create a convolutional model. A simple one will do for beginners. At this point,
           make sure it trains and improves accuracy from the data.
        5. Implement validation and testing algorithms
        6. When this pipeline is finished, work with the experimentation!

    """
    # Read csv for urbanSounds
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')

    # Initialize class
    audio_model = ClassifyAudio(df)

    # Load and preprocess training data
    x_train, y_train = audio_model.create_training_data_shuffle()

    # Compile model
    audio_model.set_up_model(x_train)

    # Train network
    audio_model.train(x_train, y_train)

    # TODO: The folds are shuffled together. They should be trained on separately.
    #       Perhaps one fold should be one batch.
    # TODO: Save features in an array


main()
