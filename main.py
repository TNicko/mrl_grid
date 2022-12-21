import gym
import numpy as np
from mrl_grid.custom_envs.grid_env import GridEnv
import mrl_grid.models as model

# Number of episodes to be run on environment
N_EPISODES = 10

# Define hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

if __name__ == "__main__":

    start_state = (0, 0) # set start state of agent

    # Initialize environment
    env = GridEnv(5, 5, start_state)

    # model.no_learning(env, N_EPISODES)
    model.q_learning(env, ALPHA, GAMMA, EPSILON, N_EPISODES)

    env.close()