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
        self.epsilon = 0.1
        session = tf.keras.backend.get_session()
        init = tf.global_variables_initializer()
        session.run(init)

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

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        print("round ", round)
        self.states_buffer.append((prev_state, prev_action, new_state, reward))
        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        board = new_state[BOARD].astype(np.float64)
        board = board.reshape((1, board.shape[0], board.shape[1], 1))

        board_ph = tf.placeholder(dtype=tf.float64, shape=[1, board.shape[1], board.shape[2], 1])
        preds = self.model(board_ph)
        i = tf.argmax(preds[0])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            index = sess.run(i, feed_dict={board_ph: board})

        return self.ACTIONS[index]

    def _q_value(self, state, action):
        if action is None:
            return float("-inf")

        board = state[BOARD].astype(np.float64)
        board = board.reshape((1, board.shape[0], board.shape[1], 1))
        pred = self.model(board)[0]
        return pred[self.actions2i[action]]

