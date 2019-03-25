
# ML libraries
from keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Dropout, Dense, LeakyReLU
from keras.models import Sequential
from keras.regularizers import l1, l2


def urban_sound_net(input_shape, num_classes, optimizer):
    """
    Network inspired by the proposed network of the creators of the Urban Sounds data set.
    :param input_shape:
    :param num_classes:
    :param optimizer:
    :return:
    """
    model = Sequential()

    # CNN - 2 - Conv
    model.add(Conv2D(24, 5, strides=1,
                     padding='same',
                     # kernel_regularizer=l2(0.001),
                     input_shape=input_shape
                     ))
    model.add(MaxPool2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(LeakyReLU())

    # CNN - 2 - Conv
    model.add(Conv2D(48, 5, strides=1,
                     # kernel_regularizer=l2(0.001),
                     padding='same'))
    model.add(MaxPool2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(LeakyReLU())

    # CNN - 3 - Conv
    model.add(Conv2D(48, 5, strides=1,
                     # kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    # CNN - 4 - FCC
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())

    # CNN - 5 Output
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001),))

    # Print summary and compile model
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


def novel_cnn(input_shape, num_classes, optimizer):
    """
    A novel convolutional model network
    :return:
    """
    model = Sequential()

    # CNN - 2 - Conv
    model.add(Conv2D(16, 5, strides=5,
                     padding='same',
                     kernel_regularizer=l2(0.001),
                     input_shape=input_shape
                     ))
    model.add(LeakyReLU())

    # CNN - 2 - Conv
    model.add(Conv2D(32, 3, strides=3,
                     kernel_regularizer=l2(0.001),
                     padding='same'))
    model.add(LeakyReLU())

    # CNN - 3 - Conv
    model.add(Conv2D(64, 2, strides=2,
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    # CNN - 4 - Conv
    model.add(Conv2D(128, (1, 1), strides=(1, 1),
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    model.add(Dropout(0.3))

    # CNN - 4 - FCC
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU())

    # CNN - 5 Output
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary and compile model
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


def novel_cnn_small_kernels(input_shape, num_classes, optimizer):
    """
    A novel convolutional model network
    :return:
    """
    model = Sequential()

    # CNN - 2 - Conv
    model.add(Conv2D(16, 1, strides=5,
                     padding='same',
                     kernel_regularizer=l2(0.001),
                     input_shape=input_shape
                     ))
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    # CNN - 2 - Conv
    model.add(Conv2D(32, 1, strides=3,
                     kernel_regularizer=l2(0.001),
                     padding='same'))
    model.add(LeakyReLU())

    # CNN - 3 - Conv
    model.add(Conv2D(64, 1, strides=2,
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    # CNN - 4 - Conv
    model.add(Conv2D(128, 1, strides=1,
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    model.add(Dropout(0.5))

    # CNN - 4 - FCC
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU())

    # CNN - 5 Output
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary and compile model
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model

def bigger_net(input_shape, num_classes, optimizer):
    """
    Network made for large data sets and more classes
    :return:
    """
    model = Sequential()

    model.add(Conv2D(16, 3, strides=3,
                     padding='same',
                     kernel_regularizer=l2(0.001),
                     input_shape=input_shape
                     ))
    model.add(LeakyReLU())

    # VGG - 1 - Conv
    model.add(Conv2D(32, 3, strides=3,
                     kernel_regularizer=l2(0.001),
                     padding='same'))
    model.add(LeakyReLU())

    # VGG - 3 - Conv
    model.add(Conv2D(64, 2, strides=2,
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    # VGG - 4 - Conv
    model.add(Conv2D(128, (1, 1), strides=(1, 1),
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    # VGG - 5 - Conv
    model.add(Conv2D(256, (1, 1), strides=(1, 1),
                     kernel_regularizer=l2(0.001),
                     padding='same', ))
    model.add(LeakyReLU())

    model.add(Dropout(0.3))

    model.add(MaxPool2D(2, 2))

    # VGG - 4 - FCC
    model.add(Flatten())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())

    # VGG - 5 Output
    model.add(Dense(num_classes, activation='softmax'))

    # Print summary and compile model
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

    return model


if __name__ == '__main__':

    model = urban_sound_net(input_shape=(128, 128, 1), num_classes=2, optimizer='adam')