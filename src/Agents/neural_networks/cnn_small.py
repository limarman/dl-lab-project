from keras import Sequential
from keras.layers import Flatten, Dense, Permute, Convolution2D, Activation


def get_cnn(input_shape: (int, int, int), num_actions: int, train_interval: int = 1):
    model = Sequential()
    # put channels last
    model.add(Permute((1, 3, 4, 2), input_shape=(train_interval,) + input_shape))
    model.add(Convolution2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    return model