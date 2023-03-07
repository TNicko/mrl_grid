import gym
import numpy as np
import os
from typing import Literal, get_args
from mrl_grid.custom_envs.grid_env import GridEnv
import mrl_grid.maps as maps
from mrl_grid.models.nna import NNA
from mrl_grid.run_models import run_ppo, run_a2c, run_dqn, load_model
import tensorflow as tf
from statistics import mean 
import datetime
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor


#TODO 
# Add multiple agents to same environment
# Add a WAIT action to action space.
# Negative rewards: collide against obstacles/other agents, no motion (waiting)
# Add max steps per epsiode ???
# Create grid environment maps and feed dynamically to GridEnv to be created
# Explore Monte Carlo Tree Search (MCTS) for RL ???

# Initialize environment
grid_name = "9x9"
grid_map = maps.SINGLE_AGENT_MAPS[grid_name]
start_pos = (0, 0) # set start state of agent
n_channels = 2       # Agent pos channel & agent path channel

episodes = 1_000_000 # Number of times environment is run
n_split = 10 # Split episode outputs into this number
render = False # Render environment

# directories
logdir = f"logs/{grid_name}"
model_dir = f"models/{grid_name}"

if __name__ == "__main__":

    env = GridEnv(grid_map, n_channels, start_pos)
    env.reset()

    model_type = input("Select type (DQN, PPO, A2C, NNA, load, test): ")

    if model_type == "PPO":
        models_dir = f"{model_dir}/PPO"
        env = Monitor(env, f"{models_dir}/")
        run_ppo(env, models_dir, logdir, episodes)

    if model_type == "A2C":
        models_dir = f"{model_dir}/A2C"
        env = Monitor(env, f"{models_dir}/")
        run_a2c(env, models_dir, logdir, episodes)
    
    if model_type == "DQN":
        models_dir = f"{model_dir}/DQN"
        print(f"{models_dir}/")
        env = Monitor(env, f"{models_dir}/")
        run_dqn(env, models_dir, logdir, episodes)

    if model_type == "test":
        model_type = input("Select type (DQN, PPO, A2C, NNA): ")
        load_steps = input("Load model at timestep: ")
        models_dir = f"{model_dir}/{model_type}"
        model_path = f"{models_dir}/{load_steps}"

        load_model(model_path, env, model_type)
    
    if model_type == "NNA" or model_type == None or model_type == "":
        render = True
        nna = NNA(env, episodes, n_split, render)
        nna.run()

    env.close()


