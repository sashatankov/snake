from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten
from tensorflow.keras import backend as bd
import tensorflow as tf
N_ACTIONS = 3


class SnakeModel(Model):

    def __init__(self):

        super(SnakeModel, self).__init__()

        self.conv1 = Conv2D(32, 3, strides=(2, 2), bias_initializer=tf.keras.initializers.Zeros())
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(64, 3, strides=(2, 2), bias_initializer=tf.keras.initializers.Zeros())
        self.relu2 = Activation('relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, bias_initializer=tf.keras.initializers.Zeros())
        self.relu3 = Activation('relu')
        self.dense2 = Dense(N_ACTIONS, bias_initializer=tf.keras.initializers.Zeros())

        session = bd.get_session()
        init = tf.global_variables_initializer()
        session.run(init)
    def __call__(self, x, *args, **kwargs):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.relu3(y)
        y = self.dense2(y)

        return y





