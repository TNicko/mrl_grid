import numpy as np

class QLearning:
    """
    Implements Q-learning algorithm.
    """
    def __init__(self, env, lr, gamma, epsilon):

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        
        self.Q = np.zeros((env.observation_space.shape[0], env.nA))

    def train(self, action, state, next_state, reward):

        # Update Q-value for the current state and action
        old_val = self.Q[state, action]
        next_max = np.max(self.Q[next_state])
        new_val = (1 - self.lr) * old_val + self.lr * (reward + self.gamma * next_max)

        self.Q[state, action] = new_val

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.Q.shape[1]) # Explore
        else:
            action = np.argmax(self.Q[state]) # Exploit

        return action

    def decay_epsilon(self, min_epsilon, decay):
        self.epsilon = max(min_epsilon, self.epsilon * decay)
