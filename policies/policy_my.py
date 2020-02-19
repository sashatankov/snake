from policies import base_policy as bp
import tensorflow as tf
import numpy as np
from policies.snake_model import SnakeModel
BOARD = 0


class MyPolicy(bp.Policy):

    def init_run(self):

        self.actions2i = {a: i for i, a in enumerate(MyPolicy.ACTIONS)}
        self.actions2i[None] = 2
        self.states_buffer = list()  # to save the state after each act() call
        self.learning_rate = 0.01
        self.discount_factor = 0.95

        self.model = SnakeModel()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam()

    def cast_string_args(self, policy_args):
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        for prev, action, new_s, r in self.states_buffer:
            if prev is None or action is None:
                continue
            with tf.GradientTape() as tape:
                pred_q = self._q_value(prev, action)
                new_s_board = new_s[BOARD].astype(np.float64)
                new_s_board = new_s_board.reshape((1, new_s_board.shape[0], new_s_board.shape[1], 1))
                target_q = r + self.discount_factor * tf.reduce_max(self.model(new_s_board)[0])
                loss = tf.square(pred_q - target_q)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.states_buffer.clear()
        self.act(round, prev_state, prev_action, reward, new_state, too_slow)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.states_buffer.append((prev_state, prev_action, new_state, reward))
        board = new_state[BOARD].astype(np.float64)
        board = board.reshape((1, board.shape[0], board.shape[1], 1))
        preds = self.model(board)
        i = tf.argmax(preds[0])

        return self.ACTIONS[i]

    def _q_value(self, state, action):
        if action is None:
            return float("-inf")
        board = state[BOARD].astype(np.float64)
        board = board.reshape((1, board.shape[0], board.shape[1], 1))
        pred = self.model(board)[0]
        return pred[self.actions2i[action]]

