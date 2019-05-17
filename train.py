
# Standard libraries
import configparser
import re

import pandas as pd
import numpy as np
import os


# ML libraries
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import keras.optimizers
import keras.backend as backend

# Miscellaneous
from tqdm import tqdm

# Personal libraries
from preprocess_data import PreprocessData
import models.ConvModels as CNN

# Parse the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")


def load_and_extract_features(filepath):
    """
    Load features from .npz file and creates an attribute with the tuple of features
    :param filepath:
    :return:
    """
    # Load and fetch features
    features = np.load(filepath, mmap_mode='r')
    return features['x'], features['y']


class UpdateClassWeight(keras.callbacks.Callback):
    """
    Callback object that prints the learning rate
    """
    def __init__(self, audio_classifier):
        super().__init__()
        self.audio_classifier = audio_classifier

    def on_epoch_end(self, epoch, logs=None):
        """

        :param batch:
        :param logs:
        :return:
        """
        print(self.audio_classifier.augs)
        self.audio_classifier.augs = {}



class TrainAudioClassificator(PreprocessData):
    """
    Class object for training a model to classify acoustic sounds
    """
    def __init__(self, df):
        """
        Initialize variables
        """
        PreprocessData.__init__(self, df)

        # Audio parameters
        self.randomize_rolls = config['model'].getboolean('randomize_roll')
        self.fold = config['model'].getint('fold')
        self.epochs = config['model'].getint('epochs')
        self.batch_size = config['model'].getint('batch_size')
        assert self.batch_size >= 1+len(self.time_shift_param)+len(self.pitch_shift_param), \
            f'batch_size must be bigger than {1+len(self.time_shift_param)+len(self.pitch_shift_param)} because' \
            f'data augmentation is used'

        self.batch_size_for_test = 32
        self.learning_rate = config['model'].getfloat('learning_rate')
        self.fold = config['model'].getint('fold')
        self.steps_per_epoch = config['model'].getint('steps_per_epoch')

        # Chooses optimizer
        self.optimizer = config['model']['optimizer']
        if self.optimizer == 'adadelta':
            self.optimizer = keras.optimizers.adadelta()
        elif self.optimizer == 'adam':
            self.optimizer = keras.optimizers.adam(lr=self.learning_rate)

        # Choose model
        self.network_mode = config['model']['network_mode']
        self.test_mode = config['model']['test_mode']
        self.network_architecture = config['model']['network_architecture']
        self.load_model_path = config['model']['load_model']
        self.load_weights = config['model'].getboolean('load_weights')

        # Parameters to be initiated at a later stage
        self.class_weight = None
        self.input_shape = None
        self.model = None
        self.features = None
        self.num_classes = None
        self.callbacks_list = None
        self.validation_fold = None

    def compute_class_weight(self, y_train):
        """
        Computes the class weight distribution which is is used in the model to compensate for over represented
        classes.
        :return:
        """
        y_flat = np.argmax(y_train, axis=1)  # Reshape one-hot encode matrix back to string labels
        self.class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        print(self.class_weight)

    def set_up_model(self, x, y):
        """
        Methods compiles the specified model. Currently only CNN is available.
        :return:
        """

        # Load a predefined network
        if self.load_weights is True:
            self.model = load_model(self.load_model_path)
            self.model.summary()

        # Define input shape and compile model
        else:
            num_classes = y.shape[1]
            if self.file_type == 'jpeg':
                input_shape = (x.shape[1], x.shape[2], 3)
            elif self.file_type =='wav':
                input_shape = (x.shape[1], x.shape[2], 1)

            self.model = CNN.fetch_network(self.network_architecture,
                                           input_shape, num_classes, self.optimizer)

        # Set up tensorboard and checkpoint monitoring
        tb_callbacks = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                   write_grads=True, write_images=False, embeddings_freq=0,
                                   embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                   update_freq=1500)

        checkpoint = ModelCheckpoint('weights/weights.{epoch:02d}-{val_acc:.2f}' +
                                     f'_{self.feature}_fold{self.validation_fold}.hdf5',
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        check = UpdateClassWeight(self)
        self.callbacks_list = [checkpoint, tb_callbacks, check]

    def train(self):
        """
        Trains the model on data and validates after every epoch.
        :return:
        """

        # Train on data extracted
        print(f"Training on: {sum(self.n_training_samples.values())} samples\n"
              f"Validating on: {sum(self.n_validation_samples.values())} samples")

        self.model.fit_generator(self.preprocess_dataset_generator(mode='training'),
                                 # steps_per_epoch=10,
                                 steps_per_epoch=int(sum(self.n_training_samples.values())/self.batch_size),
                                 epochs=self.epochs, verbose=1,
                                 callbacks=self.callbacks_list,
                                 validation_data=self.preprocess_dataset_generator(mode='validation'),
                                 validation_steps=int(sum(self.n_validation_samples.values())/self.batch_size),
                                 class_weight=self.class_weight, max_queue_size=3,
                                 # workers=2, use_multiprocessing=True,
                                 shuffle=True, initial_epoch=0)

    def test_model(self, mode='test_normal'):
        """
        Method that manually performs testing on the test set with a trained model.

        mode = test_normal: Perform a test on one given set of weights specified in the config_file
        mode = test_all: Test all the weights and return the top 5 best accuracy tests.
        :param mode:
        :return:
        """
        steps = int(sum(self.n_testing_samples.values()) / self.batch_size_for_test)
        print(f"Training on: {sum(self.n_training_samples.values())} samples\n"
              f"Validating on: {sum(self.n_validation_samples.values())} samples")
        # Goes though all the weights and returns the 5 top best accuracies
        if mode == 'test_all':

            accs = []
            conf_matrices = []
            mious = []

            weights = config['model']['weights'].replace('\n', '').split(',')[:-1]

            seed = 50

            # Loop through the saved weights
            for weight in weights:
                name_of_run = re.findall(r'.\\(.*)\\weights', weight)
                self.feature = re.findall(r"\d\d_(.*)_fold", weight)[0]
                self.audio_folder = self.feature
                model = load_model(weight)
                # Generate labels for the test set which is used to generate the test set.
                data = self.generate_labels(seed)
                y_pred = []

                # Run predictions on the test folds
                prediction = model.predict_generator(self.generate_data_for_predict_generator(data),
                                                     steps=steps,
                                                     verbose=1,
                                                     )
                # Decode from one-hot encoding
                prediction = np.argmax(prediction, axis=1)
                for pred in prediction:
                    y_pred.append(self.classes[pred])

                matrix, accuracy, miou = self.evaluate_performance_of_test(data, y_pred)
                print(name_of_run)
                print(self.classes)
                print(f'Confusion matrix:\n{matrix}')
                print(f'Accuracy: {accuracy}')
                print(f'MIoU: {miou}\n')

                backend.clear_session()
                accs.append(np.trace(matrix)/np.sum(matrix))
                conf_matrices.append(matrix)
                mious.append(miou)
            #
            # # Fetch the top best set of weights
            # indexes = np.argpartition(accs, -5)[-5:]
            # top_five_acc = np.array(accs)[indexes]
            # top_five_matrix = np.array(conf_matrices)[indexes]
            # top_five_miou = np.array(mious)[indexes]

            # for acc, matrix, miou in zip(top_five_acc, top_five_matrix, top_five_miou):
            #     print(self.classes)
            #     print(f'Confusion matrix:\n{matrix}')
            #     print(f'Accuracy: {acc}')
            #     print(f'MIoU: {miou}\n')

        # Tests one set of weights instead of all the weights
        elif mode == 'test_normal':
            # Load the model
            model = load_model(self.load_model_path)
            # Generate labels for the test set which is used to generate the test set.
            for seed in np.random.randint(0, 1000, 20):
                y_pred = []
                data = self.generate_labels(seed)
                prediction = model.predict_generator(self.generate_data_for_predict_generator(data),
                                                     steps=steps,
                                                     verbose=1
                                                     )
                # Decode from one-hot encoding
                prediction = np.argmax(prediction, axis=1)
                for pred in prediction:
                    y_pred.append(self.classes[pred])

                # Evaluate run
                matrix, accuracy = self.evaluate_performance_of_test(data, y_pred)

                print("\n")
                print(self.classes)
                print(matrix)
                print(accuracy)

    def evaluate_performance_of_test(self, data, y_pred):
        """
        Calculates confusion matrix and mean Intersect Over Union
        :return:
        """

        # Create confusion matrix calculate accuracy
        matrix = confusion_matrix(data['labels'][:len(y_pred)], y_pred, self.classes)
        accuracy = np.trace(matrix) / np.sum(matrix)

        miou = self.mean_intersect_over_union(matrix)

        return matrix, accuracy, miou

    @staticmethod
    def mean_intersect_over_union(matrix):
        """
        Calculate mean inttersect over union
        :param matrix:
        :return:
        """
        # Find mIoU
        true_positives = np.diagonal(matrix)
        miou = []
        for i, t in enumerate(true_positives):
            false_positive = matrix[:, i]
            false_positive = np.delete(false_positive, [i, i])
            false_positive = sum(false_positive)
            false_negative = matrix[i, :]
            false_negative = np.delete(false_negative, [i, i])
            false_negative = sum(false_negative)
            miou.append(t / (t + false_positive + false_negative))
        miou = sum(miou) / len(miou)
        return miou


def run(audio_model):
    """
    Set up model and start training
    :param audio_model:
    :return:
    """

    # Train a network
    if audio_model.network_mode == 'train_network':

        # Generate a sample of the data to use as setup for the model
        audio_model.batch_size = 500
        x_train, y_train = next(audio_model.preprocess_dataset_generator(mode='training'))
        audio_model.batch_size = config['model'].getint('batch_size')

        # Compile model
        audio_model.set_up_model(x_train, y_train)

        # Compute class weight
        audio_model.compute_class_weight(y_train)

        # # Train network
        audio_model.train()

    # Test a network
    elif audio_model.network_mode == 'test_network':

        # Test network
        audio_model.test_model(audio_model.test_mode)

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
    dataframe_file = config['preprocessing']['csv_file']
    df = pd.read_csv(dataframe_file)

    # Initialize class
    audio_model = TrainAudioClassificator(df)

    # Start training or predicting
    run(audio_model)


main()
