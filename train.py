
# Standard libraries
import configparser
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization, Conv2DTranspose
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import keras.optimizers

# Miscellaneous
from tqdm import tqdm
from python_speech_features import mfcc, fbank, logfbank
import soundfile as sf

# Personal libraries
import preprocess_data
import plotting_functions

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


class TrainAudioClassificator:
    """
    Class object for training a model to classify acoustic sounds
    """
    def __init__(self, df):
        """
        Initialize variables
        """
        # DataFrame
        self.df = df

        # Find classes and create class distribution
        self.classes = ['air_conditioner', 'engine_idling', 'siren', 'street_music', 'drilling']
        # self.classes = list(np.unique(df['label']))
        self.class_dist = df.groupby(['label'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()

        # Audio parameters
        self.network_type = config['model']['net']
        self.load_file = config['preprocessing']['load_file']
        self.save_file = config['preprocessing']['save_file']
        self.randomize_rolls = config['preprocessing'].getboolean('randomize_roll')
        self.feature = config['preprocessing']['feature']
        self.n_filt = config['preprocessing'].getint('n_filt')
        self.n_feat = config['preprocessing'].getint('n_feat')
        self.n_fft = config['preprocessing'].getint('n_fft')
        self.rate = config['preprocessing'].getint('rate')
        self.step = int(self.rate / config['preprocessing'].getfloat('step_size'))
        self.activate_threshold = config['preprocessing'].getboolean('activate_threshold')
        self.threshold = config['preprocessing'].getfloat('threshold')
        self.n_samples = config['preprocessing'].getint('n_samples')

        # Training parameters
        self.epochs = config['model'].getint('epochs')
        self.batch_size = config['model'].getint('batch_size')
        self.learning_rate = config['model'].getfloat('learning_rate')

        # Chooses optimizer
        self.optimizer = config['model']['optimizer']
        if self.optimizer == 'adadelta':
            self.optimizer = keras.optimizers.adadelta()
        elif self.optimizer == 'adam':
            self.optimizer = keras.optimizers.adam(lr=self.learning_rate)

        # Choose model
        self.network_mode = config['model']['network_mode']
        self.load_model_path = config['model']['load_model']
        self.load_weights = config['model'].getboolean('load_weights')

        # Parameters to be initiated at a later stage
        self.class_weight = None
        self.input_shape = None
        self.model = None
        self.features = None
        self.x = None
        self.y = None
        self.callbacks_list = None
        self.validation_fold = None

    def set_up_model(self, train_x):
        """
        Methods compiles the specified model. Currently only CNN is available.
        :return:
        """
        # Load weights if activated
        if self.load_weights is True or self.network_mode == 'test_network':
            self.model = load_model(self.load_model_path)

        # Define input shape and compile model
        else:
            self.input_shape = (train_x.shape[1], train_x.shape[2], 1)
            self.model = self.convolutional_model()

        # Set up tensorboard and checkpoint monitoring
        tb_callbacks = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                   write_grads=True, write_images=False, embeddings_freq=0,
                                   embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                   update_freq=20000)

        checkpoint = ModelCheckpoint('weights/weights.{epoch:02d}-{val_acc:.2f}'+f'_fold_{self.validation_fold}.hdf5',
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        self.callbacks_list = [checkpoint, tb_callbacks]

    def convolutional_model(self):
        """
        A novel convolutional model network
        :return:
        """
        # TODO: try defining a learning rate
        model = Sequential()

        # VGG - 1 - Conv
        model.add(Conv2D(256, (3, 3), activation='relu', strides=(1, 1),
                         padding='same', input_shape=self.input_shape))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # Upsampling
        model.add(Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', activation='relu'))
        # model.add(Conv2D(32, (3, 3), activation='relu', strides=(2, 2),
        #                  padding='same', ))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # VGG - 2 - Conv
        model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                         padding='same', ))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # Upsampling
        # model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2),
        #                  padding='same', ))
        model.add(Conv2DTranspose(32, (2, 2), strides=(1, 1), padding='same', activation='relu'))

        # VGG - 3 - Conv
        model.add(Conv2D(16, (3, 3), activation='relu', strides=(2, 2),
                         padding='same', ))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        # VGG - 4 - FCC
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        # VGG - 5 Output
        model.add(Dense(self.y.shape[-1], activation='softmax'))

        # Print summary and compile model
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['acc'])

        return model

    def build_feature_from_signal(self, file_path, feature_to_extract='mfcc', activate_threshold=False):
        """
        Reads the signal from the file path. Then build mffc features that is returned.
        If the signal is stereo, the signal will be split up and only the first channel is used.

        Later implementations should consist of using both channels, and being able to select other features than mfccs
        :param feature_to_extract: Choose what feature to extract. Current alternatives: 'mffc', 'fbank' and 'logfbank'
        :param file_path: File path to audio file
        :param activate_threshold: A lower boundary to filter out weak signals
        :return:
        """
        # Read file
        sample, rate = sf.read(file_path)
        # TODO: Find the minimum length of a signal as use that as step length
        #       Right now the smallest signals of 0.05 seconds is just skipped. Though there are only like three.

        # If len is > 1 it means the signal is stereo and needs to be split up
        if len(sample.shape) > 1:
            sample, signal_2 = preprocess_data.separate_stereo_signal(sample)

        # Choose a window of the signal to use for the sample
        try:
            rand_index = np.random.randint(0, sample.shape[0] - self.step)
            sample = sample[rand_index:rand_index + self.step]


        except ValueError:
            print("audio file to small. Skip and move to next")
            return 'move to next file'

        # Perform filtering with a threshold on the time signal
        if activate_threshold is True:
            mask = preprocess_data.envelope(sample, rate, self.threshold)
            sample = sample[mask]

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
            raise ValueError('Please choose an existing feature: mfcc, logfbank or fbank ')
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
        fold_list_x = []
        fold_list_y = []
        _min, _max = float('inf'), -float('inf')    # Initialize min and max for x
        folds = list(np.unique(self.df['fold']))    # The folds to loop through

        # Build feature samples
        for idx,  fold in enumerate(folds):
            x, y = [], []  # Set up lists
            print(f'\nExtracting data from fold{fold}')
            files_in_fold = self.df.loc[self.df.fold == fold]  # Get the filenames that exists in that fold
            # samples_per_fold = 2 * int(files_in_fold['length'].sum() / 0.1)  # the number of samples for each fold
            samples_per_fold = self.n_samples

            # Loop through the files in the fold
            for _ in tqdm(range(samples_per_fold)):

                # Pick a random class from the probability distribution and then a random file with that class
                # rand_class = np.random.choice(self.class_dist.index, p=self.prob_dist)
                rand_class = np.random.choice(self.classes)
                file = np.random.choice(files_in_fold[self.df.label == rand_class].slice_file_name)
                file_path = f'../Datasets/audio/downsampled/fold{fold}/{file}'

                # Extract feature from signal
                x_sample = self.build_feature_from_signal(file_path,
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
            x = (x - _min) / (_max - _min)                          # Normalize x and y
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)    # Reshape all to same size

            # One-hot encode the labels and append to fold
            y = to_categorical(y, num_classes=len(self.classes))                   # One hot encoding
            fold_list_x.append(x)
            fold_list_y.append(y)

        # Save the features in a .npz file
        np.savez(self.save_file, x=fold_list_x, y=fold_list_y)

    def compute_class_weight(self, y_train):
        """
        Computes the class weight distribution which is is used in the model to compensate for over represented
        classes.
        :return:
        """
        y_flat = np.argmax(y_train, axis=1)  # Reshape one-hot encode matrix back to string labels
        self.class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        # plotting_functions.plot_class_distribution(self.class_weight, self.classes)

    def create_training_data_shuffle(self):
        """
        Does the same as preprocess datasets except that the folds are mixed together and not kept separate.
        This will most likely provide a validation set that contain audio files that will have samples in both
        training and validation wich may give out wrong test results.
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
            x.append(x_sample if self.network_type == 'conv' else x_sample.T)
            y.append(self.classes.index(rand_class))

        # Normalize X and y
        y = np.array(y)
        x = np.array(x)
        x = (x - _min) / (_max - _min)

        # Reshape X based on 'conv' mode or 'time' mode
        if self.network_type == 'cnn':
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

        # One-hot encode the labels
        y = to_categorical(y, num_classes=10)

        # Computes the class weight for the samples extracted
        y_flat = np.argmax(y, axis=1)  # Reshape one-hot encode matrix back to string labels
        self.class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

        if self.network_mode is True:
            np.savez('usounds_features/test', x, y)

        return x, y

    def train(self, x_train, y_train, x_val, y_val):
        """
        Method used for training a model.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        # Train the network
        self.model.fit(x_train, y_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       callbacks=self.callbacks_list,
                       # class_weight=self.class_weight,
                       validation_data=(x_val, y_val)
                       )

    def test(self, x_test, y_test):
        """
        Method that manually performs testing on the validation set with a trained model
        :param x_test: test data
        :param y_test: test labels
        :return:
        """
        y_pred = []
        y_true = []

        # Loop through validation set
        i = 0
        for sample, label in tqdm(zip(x_test, y_test)):
            i += 1
            sample = np.resize(sample, (1,
                                        sample.shape[0],
                                        sample.shape[1],
                                        sample.shape[2]))           # Resize sample to fit model
            prediction = self.model.predict(sample)                 # Make a prediction

            y_pred.append(self.classes[np.argmax(prediction)])      # Append the prediction to the list
            y_true.append(self.classes[np.argmax(label)])
            if i == 1000:
                break

        # Create confusion matrix calculate accuracy
        matrix = confusion_matrix(y_true, y_pred, self.classes)
        accuracy = np.trace(matrix)/np.sum(matrix)
        print("\n")
        print(self.classes)
        print(matrix)
        print(accuracy)

    def choose_classes(self):
        """
        In the config file, pick what classes to train on and to test on. This method exist so
        the user can choose to only train on a few classes without having to create a whole new data set.
        :return:
        """
        classes_idx = []
        classes_to_use = ['air_conditioner', 'engine_idling', 'jackhammer']
        for c in classes_to_use:
            classes_idx.append(self.classes.index(c))

        s = np.array(self.y)
        print(s)

    def separate_loaded_data(self, nr_rolls=0):
        """
        When the UrbanSounds data is loaded, it will be loaded as a tuple of size 10.
        Each tuple contains the array with all the features extracted from that fold.
        This function concatenates the tuple into one large array.
        :param nr_rolls: The number of rolls for the folds. 5 rolls will set fold nr 5 as validation. 2 Will set fold 8.
        :return: x_train, y_train, x_test, y_test
        """

        # TODO: Change it som nr_rolls actually means to choose what fold to use.

        # Randomize the roll if set to True
        if self.randomize_rolls is True:
            nr_rolls = np.random.randint(0, len(self.x))
            print(f"Validating on fold{(len(self.x)-nr_rolls)}")

        self.validation_fold = (len(self.x)-nr_rolls)

        # If nr_rolls is specified, rotate the data set
        self.x = np.roll(self.x, nr_rolls, axis=0)
        self.y = np.roll(self.y, nr_rolls, axis=0)

        # Pick the classes to use in the training set.
        # self.choose_classes()

        # Concatenate the array
        x_train = np.concatenate(self.x[:-2], axis=0)
        y_train = np.concatenate(self.y[:-2], axis=0)
        x_val = self.x[-2]
        y_val = self.y[-2]
        x_test = self.x[-1]
        y_test = self.y[-1]

        # print(np.sum(y_train, axis=0))


        return x_train, y_train, x_test, y_test, x_val, y_val

    def load_and_extract_features(self, filepath):
        """
        Load features from .npz file and creates an attribute with the tuple of features
        :param filepath:
        :return:
        """
        # Load and fetch features
        self.features = np.load(filepath)
        self.x, self.y = self.features['x'], self.features['y']


def run(audio_model):
    """
    Set up model and start training
    :param audio_model:
    :return:
    """
    # Randomly extract features to create a data set
    if audio_model.network_mode == 'save_features':
        audio_model.preprocess_dataset()

    # Train a network
    elif audio_model.network_mode == 'train_network':

        # Load the features from a .npz file
        audio_model.load_and_extract_features(audio_model.load_file)

        # Create train and validation split
        x_train, y_train, _, _, x_val, y_val= audio_model.separate_loaded_data()

        # Compile model
        audio_model.set_up_model(x_train)

        # Compute class weight
        audio_model.compute_class_weight(y_train)

        # Train network
        audio_model.train(x_train, y_train, x_val, y_val)

    # Test a network
    elif audio_model.network_mode == 'test_network':

        # Load the features from a .npz file
        audio_model.load_and_extract_features(audio_model.load_file)

        # Create train and validation split
        _, _, x_test, y_test, _, _ = audio_model.separate_loaded_data()

        # # Compile model
        # audio_model.set_up_model(x_test)
        #
        # # Test network
        # audio_model.test(x_test, y_test)

    else:
        raise ValueError('Choose a valid Mode')


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
    # Read csv for UrbanSounds
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length.csv')

    # Initialize class
    audio_model = TrainAudioClassificator(df)

    # Start training or predicting
    run(audio_model)


main()
