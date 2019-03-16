import numpy as np
import keras
# ML libraries
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Dropout, Dense, LeakyReLU
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import keras.optimizers

class model():
    def __init__(self):
        """

        """

    def convolutional_model(self, input_shape, num_classes):
        """
        A novel convolutional model network
        :return:
        """
        model = Sequential()

        model.add(Conv2D(16, 5, strides=5,
                         padding='same',
                         kernel_regularizer=l2(0.001),
                         input_shape=input_shape
                         ))
        model.add(LeakyReLU())
        # model.add(Dropout(0.4))
        # model.add(BatchNormalization())

        # VGG - 1 - Conv
        model.add(Conv2D(32, 3, strides=3,
                         kernel_regularizer=l2(0.001),
                         padding='same'))
        model.add(LeakyReLU())
        # model.add(Dropout(0.4))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(2, 2))

        # VGG - 3 - Conv
        model.add(Conv2D(64, 2, strides=2,
                         kernel_regularizer=l2(0.001),
                         padding='same', ))
        model.add(LeakyReLU())
        # model.add(Dropout(0.4))
        # model.add(BatchNormalization())

        # VGG - 4 - Conv
        model.add(Conv2D(128, (1, 1), strides=(1, 1),
                         kernel_regularizer=l2(0.001),
                         padding='same', ))
        model.add(LeakyReLU())
        # model.add(Dropout(0.3))
        # model.add(BatchNormalization())
        # model.add(MaxPool2D(2, 2))

        # # VGG - 5 - Conv
        # model.add(Conv2D(256, (1, 1), strides=(1, 1),
        #                  kernel_regularizer=l2(0.001),
        #                  padding='same', ))
        #
        # model.add(LeakyReLU())
        # model.add(Dropout(0.3))
        # model.add(BatchNormalization())

        # # VGG - 6 - Conv
        # model.add(Conv2D(256, (1, 1), strides=(1, 1),
        #                  kernel_regularizer=l2(0.001),
        #                  padding='same', ))
        #
        # model.add(LeakyReLU())
        model.add(Dropout(0.3))
        # model.add(BatchNormalization())
        #
        # model.add(MaxPool2D(2, 2))

        # VGG - 4 - FCC
        model.add(Flatten())
        model.add(Dense(64))
        model.add(LeakyReLU())
        # model.add(Dense(128))
        # model.add(LeakyReLU())
        # model.add(Dense(128))
        # model.add(LeakyReLU())

        # VGG - 5 Output
        model.add(Dense(num_classes, activation='softmax'))

        # Print summary and compile model
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['acc'])

        return model