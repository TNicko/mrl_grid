import numpy as np
from mrl_grid.env import Env

class QLR(Env):
    """
    Implements Q-learning algorithm.
    """
    def __init__(self, env, episodes, n_split, render):
        super().__init__(env, episodes, n_split, render)
        self.lr = 0.01
        self.gamma = 0.9
        self.epsilon = 0.8
        self.min_epsilon = 0.1
        self.decay = 0.999
        
        self.nS = 2 ** (self.env.grid_size * 2) * self.env.grid_size # no of states
        self.Q = np.zeros((self.nS, env.nA))

    def run(self):
        for n in range(self.episodes):
            steps = 0
            episode_reward = 0
            done = False
            grid_state = self.env.reset()
            state = self.encode_state(grid_state)

            self.decay_epsilon()
            
            while not done:
                if self.render: self.env.render() # Render grid 

                action = self.policy(state)
                
                next_state, reward, done = self.env.step(action) # Take step in env
                next_state = self.encode_state(next_state)

                self.train(action, state, next_state, reward)

                state = next_state
                episode_reward += reward
                steps += 1

            self.total_rewards[n] = episode_reward
            self.total_steps[n] = steps
            episode_reward = round(episode_reward, 2)

            if n % self.n_split == 0 or n == self.episodes-1:
                avg_reward, avg_steps, max_steps = self.set_avg_episode(n)

                self.print_episode(n, avg_steps, avg_reward, avg=True)

    def train(self, action, state, next_state, reward):

        # Update Q-value for the current state and action
        old_val = self.Q[state, action]
        next_max = np.max(self.Q[next_state])
        new_val = (1 - self.lr) * old_val + self.lr * (reward + self.gamma * next_max)

        self.Q[state, action] = new_val

    def policy(self, state):
        """
        Select an action using an epsilon-greedy policy
        """
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.Q.shape[1]) # Explore
        else:
            action = np.argmax(self.Q[state]) # Exploit

        return action

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def encode_state(self, state) -> int:
        flat_grid = state.flatten()
        state = np.array(flat_grid, dtype=np.float32)
        """
        Convert state array to integer 
        """
        state = state.astype(int)
        state_int = int(''.join(map(str, state)), 2)
        # print(state_int)
        return state_int

