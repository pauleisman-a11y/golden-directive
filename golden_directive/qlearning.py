"""Tabular Q-learning (paper section 6: all pilot learning evidence is tabular)."""

import numpy as np


class QLearner:
    def __init__(self, n_actions=5, alpha=0.1, gamma=0.99, epsilon=0.1, rng=None):
        self.q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.n_actions = n_actions

    def _row(self, state):
        row = self.q.get(state)
        if row is None:
            row = np.zeros(self.n_actions)
            self.q[state] = row
        return row

    def act(self, state, greedy=False):
        row = self._row(state)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(row))

    def update(self, state, action, reward, next_state, done):
        row = self._row(state)
        target = reward
        if not done:
            target += self.gamma * float(np.max(self._row(next_state)))
        row[action] += self.alpha * (target - row[action])
