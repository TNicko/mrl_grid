import gym
import numpy as np
from mrl_grid.custom_envs.grid_env import GridEnv
from mrl_grid.models import DQN, no_learning, q_learning
import tensorflow as tf
from statistics import mean 
import datetime

def run(env, TrainNet, TargetNet, epsilon, copy_step):
    """
    :param env: training environment
    :param TrainNet: 
    :param TargetNet:
    :param epsilon: value (0-1) that controls trade-off between exploration & exploitation
    :param copy_step: interval steps for weight copying
    :returns rewards, mean(losses): 
    """
    rewards = 0
    count = 0
    done = False
    observations = env.reset()
    losses = list()

    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done = env.step(action)
        rewards += reward

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}

        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)

        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        
        count += 1
        if count % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    
    return rewards, mean(losses), count


def test(env, TrainNet):
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    
    while not done:
        env.render()
        action = TrainNet.get_action(observation, 0)
        observation, reward, done = env.step(action)
        steps += 1
        rewards += reward

    print("Testing results\n---------------")
    print(f"Steps: {steps}")
    print(f"Reward: {rewards}")

if __name__ == "__main__":

    # Initialize environment
    start_state = (0, 0) # set start state of agent
    width = 3
    height = 3
    env = GridEnv(width, height, start_state)

    # Define hyperparameters
    lr = 0.1         # learning rate
    gamma = 0.9      # discount factor (value between 0,1)
    copy_step = 25   # interval steps for weight copying
    num_states = len(env.observation_space.sample())
    num_actions = env.nA
    hidden_units = [10, 10]
    max_experiences = 1000
    min_experiences = 10
    batch_size = 32
    lr = 1e-2

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    episodes = 1 # Number of times environment is run
    total_rewards = np.empty(episodes)
    epsilon = 0.1 # decays overtime (value between 0,1)
    min_epsilon = 0.1
    decay = 0.999

    # n_split = 10 # Split episode outputs into this number
    # for n in range(episodes):

    #     epsilon = max(min_epsilon, epsilon * decay)

    #     total_reward, losses, steps = run(env, TrainNet, TargetNet, epsilon, copy_step)
    #     total_rewards[n] = total_reward
    #     avg_rewards = total_rewards[max(0, n-n_split):(n+1)].mean()
        
    #     if n % n_split == 0:
    #         print(f"Episode: {n} Steps: {steps} Reward: {total_reward} Epsilon: {epsilon} Avg Reward (last {n_split}): {avg_rewards} Loss: {losses}")
        
    # print(f"avg reward for last {n_split} episodes: {avg_rewards}")


    no_learning(env, episodes)
    # q_learning(env, lr, gamma, epsilon, episodes)

    env.close()

    