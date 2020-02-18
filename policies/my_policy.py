from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from policies.snake_model import SnakeModel


class MyPolicy(bp.Policy):

    def init_run(self):
        self.actions2i = {a: i for i, a in enumerate(MyPolicy.ACTIONS)}
        self.model = SnakeModel()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizer.Adam()


    def cast_string_args(self, policy_args):
        pass

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        learning_rate = 0.01
        discount_factor = 0.9
        with tf.GradientTape() as tape:
            pred_q = self._q_value(prev_state, prev_action)
            target_q = reward + discount_factor * tf.max(self.model(new_state))
            loss = self.loss_object(pred_q - target_q)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        preds = self.model(new_state)
        i = tf.argmax(preds)
        return self.ACTIONS[i]

    def _q_value(self, state, action):
        pred = self.model(state)
        return pred[self.actions2i[action]]


@tf.function
def train_step(prev_state, new_state):
    with tf.GradientTape() as tape:
        pass

