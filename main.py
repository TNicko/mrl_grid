import gym
import numpy as np
from typing import Literal, get_args
from mrl_grid.custom_envs.grid_env import GridEnv
from mrl_grid.models.dqn import DQN
from mrl_grid.models.q_learning import QLearning
from mrl_grid.models.no_learning import NoLearning
import tensorflow as tf
from statistics import mean 
import datetime

#TODO 
# Add multiple agents to same environment
# Add a WAIT action to action space.
# Negative rewards: collide against obstacles/other agents, no motion (waiting)
# Add max steps per epsiode ???
# Create grid environment maps and feed dynamically to GridEnv to be created

# Initialize environment
start_state = (0, 0) # set start state of agent
n_channels = 2       # Agent pos channel & agent path channel
width = 3
height = 3

# Define hyperparameters
lr = 0.1         # learning rate
gamma = 0.9      # discount factor (value between 0,1)
epsilon = 0.1    # decays overtime (value between 0,1)
min_epsilon = 0.1
decay = 0.99

# Number of times environment is run
episodes = 100 
n_split = 10 # Split episode outputs into this number

# Initialise lists for rewards & steps per episode
total_rewards = np.empty(episodes)
total_steps = np.empty(episodes)

TYPES = Literal["qlearning", "nolearning"]

def run(env, type: TYPES = "nolearning", render=False):
    # Check type input exists
    options = get_args(TYPES)
    assert type in options, f"'{type}' is not in {options}"

    if type == 'qlearning':
        qlr = QLearning(env, lr, gamma, epsilon)
    if type == 'nolearning':
        nol = NoLearning(env)
    
    for n in range(episodes):
        steps = 0
        episode_reward = 0
        done = False
        state = env.reset()

        if type == 'qlearning':
            qlr.decay_epsilon(min_epsilon, decay)
        
        while not done:
            if render: env.render() # Render grid 

            if type == 'qlearning':
                action = qlr.select_action(state)
            if type == 'nolearning':
                action = nol.select_action(state, type="assist")
            
            next_state, reward, done = env.step(action) # Take step in env

            if type == 'qlearning':
                qlr.train(action, state, next_state, reward)
            
            state = next_state
            episode_reward += reward
            steps += 1

        total_rewards[n] = episode_reward
        total_steps[n] = steps
        avg_reward = round(total_rewards[max(0, n-n_split):(n+1)].mean(), 2)
        avg_steps = round(total_steps[max(0, n-n_split):(n+1)].mean())
        episode_reward = round(episode_reward, 2)

        if n % n_split == 0:
            # print(f"Episode: {n} | Steps: {steps} | Reward: {episode_reward} | Avg Reward (last {n_split}): {avg_reward}")
            print("Episode: " + str(n).rjust(3) + " | avg Steps: " + str(avg_steps).rjust(4) + " | avg Reward: " + str(avg_reward).rjust(6))
        
    print("Episode: " + str(n).rjust(3) + " | avg Steps: " + str(avg_steps).rjust(4) + " | avg Reward: " + str(avg_reward).rjust(6))



if __name__ == "__main__":

    env = GridEnv(width, height, n_channels, start_state)

    run(env, type="qlearning", render=False)

    env.close()

    