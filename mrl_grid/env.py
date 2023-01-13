import numpy as np

class Env:
    def __init__(self, env, episodes, n_split, render):
        self.env = env
        self.episodes = episodes
        self.n_split = n_split
        self.render = render

        # Initialise data stores
        self.total_rewards = np.empty(episodes)
        self.total_steps = np.empty(episodes)
        self.avg_episode_data = {
            'ep': [], 
            'steps': [], 
            'steps_min': [],
            'steps_max': [],
            'reward': [], 
            'reward_min': [],
            'reward_max': []}

    def set_avg_episode(self, n):
        # Get lists for current episode split
        reward_split = self.total_rewards[max(0, n-self.n_split):(n+1)]
        steps_split = self.total_steps[max(0, n-self.n_split):(n+1)]

        avg_reward = round(reward_split.mean(), 2)
        avg_steps = round(steps_split.mean())

        self.avg_episode_data['ep'].append(n)
        self.avg_episode_data['steps'].append(avg_steps)
        self.avg_episode_data['reward'].append(avg_reward)
        self.avg_episode_data['steps_min'].append(round(min(steps_split)))
        self.avg_episode_data['steps_max'].append(round(max(steps_split)))
        self.avg_episode_data['reward_min'].append(round(min(reward_split), 2))
        self.avg_episode_data['reward_max'].append(round(max(reward_split), 2))

        return avg_reward, avg_steps, round(max(steps_split))

    def print_episode(self, n, avg_steps, avg_reward, epsilon=None, avg=True):

        data_string = "Episode: " + str(n).rjust(3)

        if avg:
            data_string += " | avg steps: " + str(avg_steps).rjust(4) 
            data_string += " | avg reward: " + str(avg_reward).rjust(6)
        else:
            data_string += " | steps: " + str(avg_steps).rjust(4) 
            data_string += " | reward: " + str(avg_reward).rjust(6)

        if epsilon != None:
            data_string += " | epsilon: " + str(epsilon).rjust(4)

        print(data_string)