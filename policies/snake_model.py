from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense



class SnakeModel(Model):

    def __init__(self):
        super(SnakeModel).__init__(self)
        self.dense1 = Dense(18, activation='relu')
        self.dense2 = Dense(9, activation='relu')
        self.dense3 = Dense(4, activation='relu')

    def __call__(self, x, *args, **kwargs):
        y = self.dense1(x)
        y = self.dense2(y)
        y = self.dense3(y)

        return y





