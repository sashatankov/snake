from policies import base_policy as bp
import tensorflow as tf
import numpy as np
from policies.snake_model import SnakeModel
BOARD = 0
tf.compat.v1.enable_eager_execution()


class Custom327337903(bp.Policy):

    def init_run(self):

        self.actions2i = {a: i for i, a in enumerate(MyPolicy.ACTIONS)}
        self.actions2i[None] = 2

        self.states_batch = None  # to save the state after each act() call
        self.action_batch = None
        self.new_states_batch = None
        self.rewards_batch = None

        self.learning_rate = 0.002
        self.discount_factor = 0.98
        self.epsilon = 5
        self.stacked_board = None
        self.model = SnakeModel()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.RMSprop()

    def cast_string_args(self, policy_args):
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        self.train_step(round, prev_state, prev_action, reward, new_state, too_slow)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if prev_action is not None and prev_state is not None:
            self.add_to_batch(round, prev_state, prev_action, reward, new_state, too_slow)
        x = (self.epsilon / (min(10000, np.ceil(round / 100.0)) * 100 + 1))
        if np.random.rand() < x:
            return np.random.choice(bp.Policy.ACTIONS)

        board = self.get_neighborhood(new_state)
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

        prev_board = self.get_neighborhood(prev_state)
        new_board = self.get_neighborhood(new_state)
        # print("prev_board shape ", prev_board.shape)
        # print("new_board shape ", new_board.shape)
        if self.states_batch is None:
            self.states_batch = list()
            self.states_batch.append(prev_board)
            self.action_batch = np.zeros(1, dtype=np.int32) + self.actions2i[prev_action]
            self.new_states_batch = list()
            self.new_states_batch.append(new_board)
            self.rewards_batch = np.zeros(1, dtype=np.int32) + reward

        else:
            self.states_batch.append(prev_board)
            self.action_batch = np.hstack((self.action_batch, np.zeros(1) + self.actions2i[prev_action]))
            self.new_states_batch.append(new_board)
            self.rewards_batch = np.hstack((self.rewards_batch, np.zeros(1) + reward))

    def train_step(self, round, prev_state, prev_action, reward, new_state, too_slow):

        #index = np.random.choice(len(self.states_batch), size=3, replace=False)
        if round % 5 == 0:
            prev = np.stack(self.states_batch)
            action = self.action_batch
            new_s = np.stack(self.new_states_batch)
            r = self.rewards_batch

            with tf.GradientTape() as tape:
                pred_q = self._q_value(prev, action)
                target_q = r + self.discount_factor * tf.reduce_max(self.model(new_s[..., np.newaxis]), axis=1)
                loss = tf.reduce_mean(tf.square(pred_q - target_q))

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # if self.states_batch.shape[2] % 50 == 0:
            self.states_batch = None
            self.action_batch = None
            self.rewards_batch = None
            self.new_states_batch = None

    def get_neighborhood(self, state):

        board, head = state
        head_pos, direction = head
        x_pos, y_pos = head_pos[0], head_pos[1]

        r, c = board.shape
        # print("in neighbor board shape ", board.shape)

        self.stacked_board = np.vstack((board, board, board))
        self.stacked_board = np.hstack((self.stacked_board, self.stacked_board, self.stacked_board))

        window = self.stacked_board[x_pos + r - 10: x_pos + r + 10, y_pos + c - 10: y_pos + c + 10].copy()

        if direction == 'N':
            return window.astype(np.float64)

        elif direction == 'E':
            window = np.rot90(window, k=3)
        elif direction == 'W':
            window = np.rot90(window, k=1)
        elif direction == 'S':
            window = np.rot90(window, k=2)

        return window.astype(np.float64)






