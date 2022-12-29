import numpy as np
import tensorflow as tf
import random

class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        # Input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (num_states,))

        # Hidden layers
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))

        # Ouput layer
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class DQN:
    def __init__(
        self, 
        num_states, 
        num_actions, 
        hidden_units, 
        gamma, 
        max_experiences, 
        min_experiences, 
        batch_size, 
        lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        print(states)
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        
        for key, value in exp.items():
            self.experience[key].append(value)
    
    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables

        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

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

            # Agent Policy
            possible_next_states = get_possible_next_states(state, env.grid_size, env.height)


            for a, s in list(possible_next_states.items()):
                pos = env.state_to_pos(s)
                if env.grid[pos[0], pos[1]] == 1:
                    possible_next_states.pop(a)

            
            if len(possible_next_states) == 0:
                action = env.action_space.sample() # Choose random available action
            else:
                action = random.choice(list(possible_next_states))


            next_state, reward, done = env.step(action) # Take step 
            episode_reward += reward  # Update the episode reward
            steps += 1

            state = next_state

        # Store the episode reward
        episode_rewards.append(episode_reward)

        print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")


def q_learning(env, lr, gamma, epsilon, n_episodes):
    """
    Implements Q-learning algorithm.
    """

    num_states = len(env.observation_space.sample())

    # Initialise the Q-table
    q_table = np.zeros((env.observation_space.shape[0], env.nA))

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
            if np.random.uniform() < epsilon:
                action = np.random.randint(q_table.shape[1]) # Explore
            else:
                action = np.argmax(q_table[state]) # Exploit

            next_state, reward, done = env.step(action) # Take step in env

            # Update Q-value for the current state and action
            old_val = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_val = (1 - lr) * old_val + lr * (reward + gamma * next_max)

            q_table[state, action] = new_val

            state = next_state
            episode_reward += reward  # Update the episode reward

            steps += 1

        print(f"Episode: {i}, Steps: {steps}, Reward: {episode_reward}")

    print(q_table)


def get_possible_next_states(state, grid_size, height):
    """
    Get list of possible next states from current state 
    """
    states = dict()

    if state % height != 0 and state > 0:
        states[0] = state - 1 # Up
    
    if state % height != height - 1:
        states[1] = state + 1 # Down

    if state >= height:
        states[2] = state - height # Left
        
    
    if state < grid_size - height:
        states[3] = state + height # Right

    return states
