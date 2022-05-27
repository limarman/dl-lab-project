from keras import Sequential
from keras.layers import Flatten, Dense, Activation


def get_mlp(input_size: int, num_actions: int, window_length: int = 4):
    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + (input_size,)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    return model


