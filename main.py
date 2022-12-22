import gym
import numpy as np
from mrl_grid.custom_envs.grid_env import GridEnv
import mrl_grid.models as model

# Number of episodes to be run on environment
N_EPISODES = 1000

# Define hyperparameters
ALPHA = 0.1 # learning rate
GAMMA = 0.5 # discount factor
EPSILON = 0.9

if __name__ == "__main__":

    start_state = (0, 0) # set start state of agent

    # Initialize environment
    env = GridEnv(5, 5, start_state)

    # model.no_learning(env, N_EPISODES)
    q_table = model.q_learning(env, ALPHA, GAMMA, EPSILON, N_EPISODES)

    env.close()