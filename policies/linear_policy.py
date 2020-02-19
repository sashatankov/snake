from policies import base_policy as bp
import numpy as np
FORWARD = 2
LEFT = 0
RIGHT = 1


class Linear(bp.Policy):

    def init_run(self):

        self.actions2i = {a: i for i, a in enumerate(Linear.ACTIONS)}
        self.actions2i[None] = 2
        self.weights = np.random.uniform(size=11)
        self.states_buffer = list()  # to save the state after each act() call
        self.learning_rate = 0.1
        self.discount_factor = 0.9

    def cast_string_args(self, policy_args):
        return policy_args

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        for prev, action, new_s, r in self.states_buffer:
            features = self.get_features(prev)
            max_q = max([self._q_value(new_s, a) for a in Linear.ACTIONS])
            max_q = r + self.discount_factor * max_q - np.dot(features[self.actions2i[action], :], self.weights)
            rate = max_q * self.learning_rate
            self.weights -= rate * features[self.actions2i[action], :]

        self.states_buffer.clear()
        self.act(round, prev_state, prev_action, reward, new_state, too_slow)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.states_buffer.append((prev_state, prev_action, new_state, reward))
        q_max = float("-inf")
        q_max_action = Linear.DEFAULT_ACTION
        for action in Linear.ACTIONS:
            q_val = self._q_value(new_state, action)
            if q_val > q_max:
                q_max = q_val
                q_max_action = action

        return q_max_action

    def _q_value(self, state, action):
        if action is None:
            return float("-inf")

        features = self.get_features(state)

        return np.dot(features[self.actions2i[action], :], self.weights)

    def get_features(self, state):
        feature_matrix = np.zeros((3, 11))
        if state is None:
            return feature_matrix

        neighborhood = self.get_neighborhood(state)
        feature_matrix[FORWARD, neighborhood[FORWARD] + 1] = 1
        feature_matrix[LEFT, neighborhood[LEFT] + 1] = 1
        feature_matrix[RIGHT, neighborhood[RIGHT] + 1] = 1

        return feature_matrix

    def get_neighborhood(self, state):
        """
        returns a 3x3 neighborhood of the position of the snake-head, as a
        flattened vector
        :param state:
        :return:
        """
        neighborhood = np.zeros(3, dtype=np.int32)
        board, head = state
        head_pos, direction = head
        x_pos, y_pos = head_pos[0], head_pos[1]

        if direction == 'N':
            neighborhood[FORWARD] = board[(x_pos - 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
            neighborhood[LEFT] = board[x_pos % head_pos.board_size[0], (y_pos - 1) % head_pos.board_size[1]]
            neighborhood[RIGHT] = board[x_pos % head_pos.board_size[0], (y_pos + 1) % head_pos.board_size[1]]
        elif direction == 'E':
            neighborhood[FORWARD] = board[x_pos % head_pos.board_size[0], (y_pos + 1) % head_pos.board_size[1]]
            neighborhood[LEFT] = board[(x_pos - 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
            neighborhood[RIGHT] = board[(x_pos + 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
        elif direction == 'S':
            neighborhood[FORWARD] = board[(x_pos + 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
            neighborhood[LEFT] = board[x_pos % head_pos.board_size[0], (y_pos + 1) % head_pos.board_size[1]]
            neighborhood[RIGHT] = board[(x_pos - 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
        elif direction == 'W':
            neighborhood[FORWARD] = board[x_pos % head_pos.board_size[0], (y_pos - 1) % head_pos.board_size[1]]
            neighborhood[LEFT] = board[(x_pos + 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
            neighborhood[RIGHT] = board[(x_pos - 1) % head_pos.board_size[0], y_pos % head_pos.board_size[1]]
        else:
            pass

        return neighborhood



