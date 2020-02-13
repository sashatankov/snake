from policies import base_policy as bp
import numpy as np


class Linear(bp.Policy):


    def init_run(self):

        self.weights = np.random.normal(size=12)

        # each action is represented by a one-hot-vector
        self.actions_one_hot_vectors = dict()
        for i, action in enumerate(Linear.ACTIONS):
            self.actions_one_hot_vectors[action] = np.zeros(len(Linear.ACTIONS))
            self.actions_one_hot_vectors[action][i] = 1

        self.states_buffer = list()  # to save the state after each act() call

    def cast_string_args(self, policy_args):
        pass

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        learning_rate = 0.01
        discount_factor = 0.9

        for prev, action, new_s, r in self.states_buffer:
            features = self.get_features(prev, action)
            max_q = max([self._q_value(new_s, a) for a in Linear.ACTIONS])
            max_q = r + discount_factor * max_q - np.dot(features, self.weights)
            rate = max_q * learning_rate
            self.weights -= rate * features

        self.states_buffer.clear()

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.states_buffer.append((prev_state, prev_action, new_state, reward))
        q_max = 0
        q_max_action = Linear.DEFAULT_ACTION
        for action in Linear.ACTIONS:
            q_val = self._q_value(new_state, action)
            if q_val > q_max:
                q_max = q_val
                q_max_action = action

        return q_max_action

    def _q_value(self, state, action):
        action_one_hot = self.actions_one_hot_vectors[action]
        neighborhood = self.get_neighborhood(state)
        features = np.hstack(neighborhood, action_one_hot)

        return np.dot(features, self.weights)

    def get_features(self, state, action):
        action_one_hot = self.actions_one_hot_vectors[action]
        neighborhood = self.get_neighborhood(state)
        features = np.hstack(neighborhood, action_one_hot)

        return features

    def get_neighborhood(self, state):
        """
        returns a 3x3 neighborhood of the position of the snake-head, as a
        flattened vector
        :param state:
        :return:
        """
        window = np.zeros((3,3))
        board, head = state
        head_pos, direction = head
        x_pos, y_pos = head_pos[0], head_pos[1]
        window[0, 0] = board[(x_pos - 1) % self.board_size, (y_pos - 1) % self.board_size]
        window[0, 1] = board[(x_pos - 1) % self.board_size, y_pos % self.board_size]
        window[0, 2] = board[(x_pos - 1) % self.board_size, (y_pos + 1) % self.board_size]
        window[1, 0] = board[x_pos % self.board_size, (y_pos - 1) % self.board_size]
        window[1, 1] = board[x_pos % self.board_size, y_pos % self.board_size]
        window[1, 2] = board[x_pos % self.board_size, (y_pos + 1) % self.board_size]
        window[2, 0] = board[(x_pos + 1) % self.board_size, (y_pos - 1) % self.board_size]
        window[2, 1] = board[(x_pos + 1) % self.board_size, y_pos % self.board_size]
        window[2, 2] = board[(x_pos + 1) % self.board_size, (y_pos + 1) % self.board_size]

        window = window.flatten()
        return window



