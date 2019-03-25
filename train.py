
# Standard libraries
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
from sklearn.metrics import confusion_matrix
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import keras.optimizers

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
        self.load_feature = config['model']['load_features']
        self.randomize_rolls = config['model'].getboolean('randomize_roll')
        self.fold = config['model'].getint('fold')
        self.epochs = config['model'].getint('epochs')
        self.batch_size = config['model'].getint('batch_size')
        self.learning_rate = config['model'].getfloat('learning_rate')
        self.fold = config['model'].getint('fold')
        self.steps_per_epoch = config['model'].getint('steps_per_epoch')
        self.use_generator = config['model'].getboolean('use_generator')

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

    def set_up_model(self, x, y):
        """
        Methods compiles the specified model. Currently only CNN is available.
        :return:
        """
        # Load weights if activated
        if self.load_weights is True or self.network_mode == 'test_network':
            self.model = load_model(self.load_model_path)
            self.model.summary()

        # Define input shape and compile model
        else:
            num_classes = y.shape[1]
            input_shape = (x.shape[1], x.shape[2], 1)
            self.model = CNN.novel_cnn(input_shape, num_classes, self.optimizer)

        # Set up tensorboard and checkpoint monitoring
        tb_callbacks = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                   write_grads=True, write_images=False, embeddings_freq=0,
                                   embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                   update_freq=500)
        # TODO: check val_loss instead of val_acc
        checkpoint = ModelCheckpoint('weights/weights.{epoch:02d}-{val_acc:.2f}'+f'_{self.feature}_fold{self.validation_fold}.hdf5',
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)
        self.callbacks_list = [checkpoint, tb_callbacks]

    def train(self, x_train, y_train, x_val, y_val):
        """
        Trains the model on data and validates after every epoch.
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        # keras.Sequential.fit()

        # Train on pre-made features
        if self.use_generator is False:
            self.model.fit(x_train, y_train,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           shuffle=True,
                           callbacks=self.callbacks_list,
                           class_weight=self.class_weight,
                           validation_data=(x_val, y_val)
                           )

        # Train on generated data batch-by-batch. Better for memory
        else:
            self.model.fit_generator(self.preprocess_dataset_generator(mode='training'),
                                     steps_per_epoch=int(sum(self.n_training_samples.values())/self.batch_size),
                                     epochs=self.epochs, verbose=1,
                                     callbacks=self.callbacks_list,
                                     validation_data=self.preprocess_dataset_generator(mode='validation'),
                                     validation_steps=int(sum(self.n_validation_samples.values())/self.batch_size),
                                     class_weight=None, max_queue_size=1,
                                     # workers=2, use_multiprocessing=True,
                                     shuffle=True, initial_epoch=0)

    def test_model(self, x_test, y_test):
        """
        Method that manually performs testing on the validation set with a trained model
        :param x_test: test data
        :param y_test: test labels
        :return:
        """
        y_pred = []
        y_true = []

        # Sequential.predict_generator()
        if self.use_generator is True:
            # generate labels for the test set which is used to generate the test set.
            y_true = self.generate_labels()
            prediction = self.model.predict_generator(self.generate_data_for_predict_generator(y_true),
                                                      steps=int(sum(self.n_testing_samples.values())/self.batch_size)

                                                      )
            # Decode from one-hot encoding
            prediction = np.argmax(prediction, axis=1)
            for pred in prediction:
                y_pred.append(self.classes[pred])

        # Uses a pre-loaded data set
        else:
            # Loop through validation set
            for sample, label in tqdm(zip(x_test, y_test)):
                sample = np.resize(sample, (1,
                                            sample.shape[0],
                                            sample.shape[1],
                                            sample.shape[2]))           # Resize sample to fit model
                prediction = self.model.predict(sample)                 # Make a prediction

                y_pred.append(self.classes[np.argmax(prediction)])      # Append the prediction to the list
                y_true.append(self.classes[np.argmax(label)])

        # Create confusion matrix calculate accuracy
        matrix = confusion_matrix(y_true[:len(y_pred)], y_pred, self.classes)
        accuracy = np.trace(matrix)/np.sum(matrix)
        print("\n")
        print(self.classes)
        print(matrix)
        print(accuracy)

    def separate_loaded_data(self, nr_rolls=0):
        """
        When the UrbanSounds data is loaded, it will be loaded as a tuple of size 10.
        Each tuple contains the array with all the features extracted from that fold.
        This function concatenates the tuple into one large array.
        :param y:
        :param x:
        :param nr_rolls: The number of rolls for the folds. 5 rolls will set fold nr 5 as validation. 2 Will set fold 8.
        :return: x_train, y_train, x_test, y_test
        """
        # Load the features from a .npz file
        x, y = load_and_extract_features(self.load_feature)

        # Randomize the roll if set to True
        if self.randomize_rolls is True:
            nr_rolls = np.random.randint(0, len(x))
            print(f"Validating on fold{(len(x)-nr_rolls)}")

        self.validation_fold = (len(x)-nr_rolls)

        # If nr_rolls is specified, rotate the data set
        x = np.roll(x, nr_rolls, axis=0)
        y = np.roll(y, nr_rolls, axis=0)

        # Concatenate the array
        x_train = np.concatenate(x[:-2], axis=0)
        y_train = np.concatenate(y[:-2], axis=0)
        x_val = x[-2]
        y_val = y[-2]
        x_test = x[-1]
        y_test = y[-1]

        return x_train, y_train, x_test, y_test, x_val, y_val


def run(audio_model):
    """
    Set up model and start training
    :param audio_model:
    :return:
    """
    # Randomly extract features to create a data set
    if audio_model.network_mode == 'save_features':
        if audio_model.random_extraction is True:
            audio_model.preprocessing_dataset_randomly()
        else:
            audio_model.preprocess_dataset()

    # Train a network
    elif audio_model.network_mode == 'train_network':

        # 1. Use pre-created features
        if audio_model.use_generator is False:

            # Create train and validation split
            x_train, y_train, _, _, x_val, y_val = audio_model.separate_loaded_data(nr_rolls=audio_model.fold)

            # Compile model
            audio_model.set_up_model(x_train, y_train)

            # Compute class weight
            audio_model.compute_class_weight(y_train)

            # Train network
            audio_model.train(x_train, y_train, x_val, y_val)

        # 2. Or use generator for each batch. This option is more memory friendly
        else:
            x_train, y_train = next(audio_model.preprocess_dataset_generator(mode='training'))

            # Compile model
            audio_model.set_up_model(x_train, y_train)

            # Compute class weight
            audio_model.compute_class_weight(y_train)

            # Train network
            audio_model.train(None, None, None, None)

    # Test a network
    elif audio_model.network_mode == 'test_network':

        # Load data set if not using the generator
        if audio_model.use_generator is not True:
            # Create train and validation split
            _, _, x_test, y_test, _, _ = audio_model.separate_loaded_data(nr_rolls=audio_model.fold)

        # Compile model
        audio_model.set_up_model(None, None)

        # Test network
        audio_model.test_model(None, None)

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
    df = pd.read_csv('../Datasets/UrbanSound8K/metadata/UrbanSound8K_length_NewTest.csv')

    # Initialize class
    audio_model = TrainAudioClassificator(df)

    # Start training or predicting
    run(audio_model)


main()
