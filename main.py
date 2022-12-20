import gym
from mrl_grid.custom_envs.grid_env import GridEnv

# Initialize environment
env = GridEnv(5, 5)
n_episodes = 10

# Initialize a list to store the rewards for each episode
episode_rewards = []

# Loop over number of episodes
for i in range(n_episodes):

    # Reset environment
    state = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    # Take a series of actions using the agent
    while not done:
        action = env.action_space.sample()
        state, reward, done = env.step(action)
        episode_reward += reward  # Update the episode reward
        steps += 1
        env.render()

    # Store the episode reward
    episode_rewards.append(episode_reward)

    print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")

env.close()