from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from policies.snake_model import SnakeModel


class MyPolicy(bp.Policy):

    def init_run(self):
        self.actions = {a: i for i, a in enumerate(MyPolicy.ACTIONS)}
        self.model = SnakeModel()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizer.Adam()


    def cast_string_args(self, policy_args):
        pass

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # TODO train_step() call goes here
        pass

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # TODO run the state thorough the network and argmax of the result
        pass


@tf.function
def train_step(prev_state, new_state):
    with tf.GradientTape() as tape:
        pass

