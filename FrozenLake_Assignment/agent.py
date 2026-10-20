import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995, seed=42):
        self.q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        return int(np.argmax(self.q[state]))

    def learn(self, state, action, reward, next_state, done):
        target = reward + (0 if done else self.gamma * np.max(self.q[next_state]))
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
