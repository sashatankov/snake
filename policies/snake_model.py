from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten
N_ACTIONS = 3


class SnakeModel(Model):

    def __init__(self):

        super(SnakeModel, self).__init__()
        # TODO to not change the architecture for now...
        self.conv1 = Conv2D(32, 3, strides=(2, 2))
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(32, 3, strides=(2, 2))
        self.relu2 = Activation('relu')
        self.flatten = Flatten()
        self.dense1 = Dense(256)
        self.relu3 = Activation('relu')
        self.dense2 = Dense(N_ACTIONS, activation='sigmoid')

    def __call__(self, x, *args, **kwargs):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.flatten(y)
        y = self.dense1(y)
        y = self.relu3(y)
        # y = self.dropout1(y)
        y = self.dense2(y)

        return y





