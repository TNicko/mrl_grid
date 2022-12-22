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
    # Initialize a list to store the rewards for each episode
    episode_rewards = []

    # Initialise the Q-table
    Q = np.zeros((env.height, env.width, env.nA))

    # visited = np.zeros((env.width, env.height), dtype=bool)

    # Loop over number of episodes
    for i in range(n_episodes):

        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # Take a series of actions using the agent
        while not done:
            # env.render()

            # Select an action using an epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Choose random available action
            else:
                action = np.argmax(Q[state[0]][state[1]]) # Choose action with highest Q-value

            next_state, reward, done = env.step(action) # Take step in env

            # Calculate new Q-value
            q_value = Q[state[0], state[1], action]
            expected_reward = reward + gamma * np.max(Q[next_state[0], next_state[1]])
            new_q_value = q_value + alpha * (expected_reward - q_value)

            # Update Q-value for the current state and action
            Q[state[0], state[1], action] = new_q_value

            state = next_state
            
            episode_reward += reward  # Update the episode reward
            steps += 1

        print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")

    return Q


