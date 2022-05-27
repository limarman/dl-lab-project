from keras import Sequential
from keras.layers import Flatten, Dense, Activation


def get_mlp(input_size: int, num_actions: int, train_interval: int = 1):
    model = Sequential()
    model.add(Flatten(input_shape=(train_interval,) + (input_size,)))
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


