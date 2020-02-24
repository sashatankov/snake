from policies import base_policy as bp
import tensorflow as tf
import numpy as np
from policies.snake_model import SnakeModel
BOARD = 0
tf.enable_eager_execution()


class MyPolicy(bp.Policy):

    def init_run(self):

        self.actions2i = {a: i for i, a in enumerate(MyPolicy.ACTIONS)}
        self.actions2i[None] = 2

        self.states_batch = None  # to save the state after each act() call
        self.action_batch = None
        self.new_states_batch = None
        self.rewards_batch = None

        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1

        self.model = SnakeModel()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam()

    def cast_string_args(self, policy_args):
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        prev = self.states_batch
        action = self.action_batch
        new_s = self.new_states_batch
        r = self.rewards_batch
        prev = prev.reshape((prev.shape[2], prev.shape[1], prev.shape[0]))
        new_s = new_s.reshape((new_s.shape[2], new_s.shape[1], new_s.shape[0]))

        with tf.GradientTape() as tape:
            pred_q = self._q_value(prev, action)
            target_q = r + self.discount_factor * tf.reduce_max(self.model(new_s[..., np.newaxis]), axis=1)
            loss = tf.reduce_mean(tf.square(pred_q - target_q))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.states_batch = None
        self.action_batch = None
        self.rewards_batch = None
        self.new_states_batch = None

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if prev_action is not None and prev_state is not None:
            self.add_to_batch(round, prev_state, prev_action, reward, new_state, too_slow)

        if np.random.rand() < self.epsilon:
            return np.random.choice(bp.Policy.ACTIONS)

        board = new_state[BOARD].astype(np.float64)
        board = board.reshape((1, board.shape[0], board.shape[1], 1))

        preds = self.model(board)
        i = tf.argmax(preds[0])
        index = i.numpy()

        return self.ACTIONS[index]

    def _q_value(self, state, action):
        if action is None:
            return float("-inf")

        state = state.copy()[..., np.newaxis]
        action = action.astype(np.int32)
        pred = self.model(state)
        pred = pred.numpy()

        return pred[:, action]

    def add_to_batch(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if self.states_batch is None:
            self.states_batch = prev_state[BOARD].astype(np.float64)
            self.action_batch = np.zeros(1, dtype=np.int32) + self.actions2i[prev_action]
            self.new_states_batch = new_state[BOARD].astype(np.float64)
            self.rewards_batch = np.zeros(1, dtype=np.int32) + reward

        else:
            self.states_batch = np.dstack((self.states_batch, prev_state[BOARD].astype(np.float64)))
            self.action_batch = np.hstack((self.action_batch, np.zeros(1) + self.actions2i[prev_action]))
            self.new_states_batch = np.dstack((self.new_states_batch, new_state[BOARD].astype(np.float64)))
            self.rewards_batch = np.hstack((self.rewards_batch, np.zeros(1) + reward))


