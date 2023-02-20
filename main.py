import gym
import numpy as np
import os
from typing import Literal, get_args
from mrl_grid.custom_envs.grid_env import GridEnv
# from mrl_grid.models.dqn import DQN
# from mrl_grid.models.qlr import QLR
from mrl_grid.models.nna import NNA
# from mrl_grid.models.ppo import Actor, Critic, Agent 
from mrl_grid.run_models import run_ppo, run_a2c
import mrl_grid.analysis as analysis
import tensorflow as tf
from statistics import mean 
import datetime
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy


#TODO 
# Add multiple agents to same environment
# Add a WAIT action to action space.
# Negative rewards: collide against obstacles/other agents, no motion (waiting)
# Add max steps per epsiode ???
# Create grid environment maps and feed dynamically to GridEnv to be created

# Initialize environment
start_pos = (0, 0) # set start state of agent
n_channels = 2       # Agent pos channel & agent path channel
width = 3
height = 3

episodes = 100_000 # Number of times environment is run
n_split = 10 # Split episode outputs into this number
render = False # Render environment


def load_model(model_path, env):
    model = PPO.load(model_path, env=env)
    
    episodes = 5
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action, states = model.predict(obs)
            obs, reward, done, info = env.step(action)

if __name__ == "__main__":

    env = GridEnv(width, height, n_channels, start_pos)
    env.reset()

    logdir = "logs/3x3"

    model_type = input("Select type (DQN, PPO, A2C, NNA or load): ")

    if model_type == "PPO":
        models_dir = "models/3x3/PPO"
        run_ppo(env, models_dir, logdir, episodes)

    if model_type == "A2C":
        models_dir = "models/3x3/A2C"
        run_a2c(env, models_dir, logdir, episodes)
    
    if model_type == "DQN":
        models_dir = "models/3x3/DQN"
        # DQN function here ...

    if model_type == "load":
        model_type = input("Select type (DQN, PPO, A2C, NNA): ")
        load_steps = input("Load model at timestep: ")
        models_dir = f"models/3x3/{model_type}"
        model_path = f"{models_dir}/{load_steps}"

        load_model(model_path, env)

    env.close()

