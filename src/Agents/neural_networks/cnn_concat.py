from keras import Sequential
from keras.layers import Flatten, Dense, Permute, Convolution2D, Activation

#TODO rebuild architecture so that scalar values can be concatenated to the dense layer in the end
def get_cnn(input_shape: (int, int, int), num_actions: int, train_interval: int = 1):
    model = Sequential()
    # put channels last
    model.add(Permute((1, 3, 4, 2), input_shape=(train_interval,) + input_shape))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    return model