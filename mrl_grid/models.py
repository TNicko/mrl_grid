import numpy as np

def no_learning(env, n_episodes):
    """
    Runs environment with no training (random actions). Used to visualise the environment 
    and not for training purposes.
    """
    
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
            env.render()

            action = env.action_space.sample() # Choose random available action
            next_state, reward, done = env.step(action) # Take step 
            episode_reward += reward  # Update the episode reward
            steps += 1

            state = next_state

        # Store the episode reward
        episode_rewards.append(episode_reward)

        print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")


def q_learning(env, alpha, gamma, epsilon, n_episodes):
    """
    Implements Q-learning algorithm.
    """

    # Initialise the Q-table
    q_table = np.zeros((env.height, env.width, env.nA))

    # Loop over number of episodes
    for i in range(n_episodes):

        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # Take a series of actions using the agent
        while not done:
            env.render()

            # Select an action using an epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(env.nA) # Explore
            else:
                action = np.argmax(q_table[state[0], state[1]]) # Exploit

            next_state, reward, done = env.step(action) # Take step in env

            # Update Q-value for the current state and action
            q_table[state[0], state[1], action] += alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])
            
            state = next_state
            
            steps += 1

        print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")

    return q_table


