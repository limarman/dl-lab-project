import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense, Activation


class MLP(tf.keras.Sequential):

    def __init__(self, input_size: int, num_actions: int, train_interval: int = 1):
        super().__init__()
        super().add(Flatten(input_shape=(train_interval,) + (input_size,)))
        super().add(Dense(1024))
        super().add(Activation('relu'))
        super().add(Dense(1024))
        super().add(Activation('relu'))
        super().add(Dense(1024))
        super().add(Activation('relu'))
        super().add(Dense(1024))
        super().add(Activation('relu'))
        super().add(Dense(1024))
        super().add(Activation('relu'))
        super().add(Dense(num_actions))
        super().add(Activation('linear'))

