from mrl_grid.env import Env

class NNA(Env):
    def __init__(self, env, episodes, n_split, render):
        super().__init__(env, episodes, n_split, render)

    def run(self):
        for n in range(self.episodes):
            steps = 0
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            while not done:
                if self.render: self.env.render() # Render grid 

                action = self.select_action(state)
                
                next_state, reward, done = self.env.step(action) # Take step in env
                
                state = next_state
                episode_reward += reward
                steps += 1

            self.total_rewards[n] = episode_reward
            self.total_steps[n] = steps
            episode_reward = round(episode_reward, 2)

            if n % self.n_split == 0 or n == self.episodes-1:

                avg_reward, avg_steps, max_steps = self.set_avg_episode(n)

                self.print_episode(n, avg_steps, avg_reward, avg=True)

    def select_action(self, state):

        action = self.env.action_space.sample() # Choose random available action

        return action


