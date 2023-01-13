import gym
import numpy as np
from typing import Literal, get_args
from mrl_grid.custom_envs.grid_env import GridEnv
from mrl_grid.models.dqn import DQN
from mrl_grid.models.qlr import QLR
from mrl_grid.models.nna import Random
import mrl_grid.analysis as analysis
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
start_pos = (0, 0) # set start state of agent
n_channels = 2       # Agent pos channel & agent path channel
width = 5
height = 5

episodes = 50 # Number of times environment is run
n_split = 10 # Split episode outputs into this number
render = False # Render environment

if __name__ == "__main__":

    env = GridEnv(width, height, n_channels, start_pos)

    dqn = DQN(env, episodes, n_split, render)
    dqn.train()

    # nol = Random(env, episodes, n_split, render)
    # nol.run()

    # analysis.plot_avg_rewards(avg_episode_data)
    # analysis.plot_rewards(episodes, dqn.total_rewards)
    # analysis.plot_steps(episodes, dqn.total_steps)

    env.close()

