import gym
import numpy as np
from typing import Literal, get_args
from mrl_grid.custom_envs.grid_env import GridEnv
from mrl_grid.models.dqn import DQNAgent, ReplayBuffer
from mrl_grid.models.q_learning import QLearning
from mrl_grid.models.no_learning import NoLearning
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

# Define hyperparameters
lr = 0.001         # learning rate
gamma = 0.9      # discount factor (value between 0,1)
epsilon = 0.1    # decays overtime (value between 0,1)
min_epsilon = 0.1
decay = 0.99

# Number of times environment is run
episodes = 500
n_split = 10 # Split episode outputs into this number

# Initialise lists for rewards & steps per episode
total_rewards = np.empty(episodes)
total_steps = np.empty(episodes)
avg_episode_data = {
    'ep': [], 
    'steps': [], 
    'steps_min': [],
    'steps_max': [],
    'reward': [], 
    'reward_min': [],
    'reward_max': []}

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
        grid_state = env.reset()
        state = encode_state(grid_state)
        
        if type == 'qlearning' or type == 'dqn':
            qlr.decay_epsilon(min_epsilon, decay)
        
        while not done:
            if render: env.render() # Render grid 

            if type == 'qlearning':
                action = qlr.select_action(state)
            if type == 'nolearning':
                action = nol.select_action(state, type="assist")
            
            next_state, reward, done = env.step(action) # Take step in env
            next_state = encode_state(next_state)

            if type == 'qlearning':
                qlr.train(action, state, next_state, reward)
            
            state = next_state
            episode_reward += reward
            steps += 1

        total_rewards[n] = episode_reward
        total_steps[n] = steps
        episode_reward = round(episode_reward, 2)

        if n % n_split == 0 or n == episodes-1:

            avg_reward, avg_steps, max_steps = set_avg_episode(n)

            print("Episode: " + str(n).rjust(3) + " | avg steps: " + str(avg_steps).rjust(4) + " | max steps: " + str(max_steps).rjust(4) + " | avg reward: " + str(avg_reward).rjust(6))

def set_avg_episode(n):
    # Get lists for current episode split
    reward_split = total_rewards[max(0, n-n_split):(n+1)]
    steps_split = total_steps[max(0, n-n_split):(n+1)]

    avg_reward = round(reward_split.mean(), 2)
    avg_steps = round(steps_split.mean())

    avg_episode_data['ep'].append(n)
    avg_episode_data['steps'].append(avg_steps)
    avg_episode_data['reward'].append(avg_reward)
    avg_episode_data['steps_min'].append(round(min(steps_split)))
    avg_episode_data['steps_max'].append(round(max(steps_split)))
    avg_episode_data['reward_min'].append(round(min(reward_split), 2))
    avg_episode_data['reward_max'].append(round(max(reward_split), 2))

    return avg_reward, avg_steps, round(max(steps_split)) 

def encode_state(state) -> int:
    flat_grid = state.flatten()
    state = np.array(flat_grid, dtype=np.float32)
    """
    Convert state array to integer 
    """
    state = state.astype(int)
    state_int = int(''.join(map(str, state)), 2)
    # print(state_int)
    return state_int

def run_env(env, agent, buffer, render):
    """
    Runs the "env" with the instructions 
    produced by "agent" and collects experiences
    into "buffer" for later training.
    """
    steps = 0
    episode_reward = 0
    done = False
    state = env.reset()

    # Decay epsilon
    agent.decay_epsilon()
    
    while not done:
        if render: env.render() # Render grid

        action = agent.policy(state)
        # print(action)
        next_state, reward, done = env.step(action) # Take step in env
        
        exp = {'s': state, 'a': action, 'r': reward, 's2': next_state, 'd': done}
        buffer.add_experience(exp)

        state = next_state
        episode_reward += reward
        steps += 1

    return episode_reward, steps

def train_dqn(env, episodes, n_split, render):
    """
    Trains a DQN agent on the environment
    """
    agent = DQNAgent(env.observation_space, env.nA)
    buffer = ReplayBuffer()

    for n in range(episodes):
        episode_reward, steps = run_env(env, agent, buffer, render)
        exp_batch = buffer.sample_batch()
        loss = agent.train(exp_batch)

        if n % 10 == 0:
            agent.update_target_network()

        total_rewards[n] = episode_reward
        total_steps[n] = steps
        episode_reward = round(episode_reward, 2)

        if n % n_split == 0 or n == episodes-1:

            avg_reward, avg_steps, max_steps = set_avg_episode(n)

            print("Episode: " + str(n).rjust(3) + " | avg steps: " + str(avg_steps).rjust(4) + " | avg reward: " + str(avg_reward).rjust(6) + " | epsilon: " + str(agent.epsilon).rjust(4))


if __name__ == "__main__":

    env = GridEnv(width, height, n_channels, start_pos)
    model = "qlearning"
    render = False
    
    train_dqn(env, episodes, n_split, render)
    # run(env, type=model, render=render)

    # analysis.plot_avg_rewards(avg_episode_data)
    analysis.plot_rewards(episodes, total_rewards)
    analysis.plot_steps(episodes, total_steps)

    env.close()

