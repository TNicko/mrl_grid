import random
import numpy as np
import tensorflow as tf
from collections import deque




class DQNAgent:
    """
    DQN Agent

    The agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """

    def __init__(self, observation_space, nA):
        # Initialise parameters
        self.nA = nA
        self.obs_space = observation_space
        self.hidden_units = [64, 64]
        self.lr = 0.01      # learning rate
        self.gamma = 0.9    # discount factor (value between 0,1)
        self.epsilon = 0.9  # decays overtime (value between 0,1)
        self.min_epsilon = 0.1
        self.decay = 0.999
        self.optimizer = tf.optimizers.Adam(self.lr)
        self.TrainNet = self._build_model()
        self.TargetNet = self._build_model()

    def _build_model(self):
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.

        :return: Q network
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape = self.obs_space.shape))

        for i in self.hidden_units:
            model.add(tf.keras.layers.Dense(i, activation='relu'))

        model.add(tf.keras.layers.Dense(self.nA, activation='linear'))
        model.compile(optimizer=self.optimizer, loss='mse')

        return model

    def policy(self, state):
        """
        Takes a state from the game environment and returns
        an action that should be taken given the current game
        environment.

        :param state: the current game environment state
        :return: an action
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.nA)
        else:
            state_input = tf.convert_to_tensor(np.expand_dims(state, axis=0))
            qs = self.TrainNet(state_input)
            action = np.argmax(qs.numpy()[0], axis=0)
            # print(qs)
            return action

    def train(self, batch):
        """
        Takes a batch of gameplay experiences from replay
        buffer and train the underlying model with the batch.
        
        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        s_batch, ns_batch, a_batch, r_batch, d_batch = batch
        current_q = self.TrainNet(s_batch)
        target_q = np.copy(current_q)
        next_q = self.TargetNet(ns_batch)
        max_next_q = np.amax(next_q, axis=1)

        for i in range(s_batch.shape[0]):
            target_q[i][a_batch[i]] = r_batch[i] if d_batch[i] else r_batch[i]+self.gamma*max_next_q[i]
        # print('current_q >>>', current_q)
        # print('target_q >>>', target_q)
        result = self.TrainNet.fit(x=s_batch, y=target_q)
        return result.history['loss']

    def update_target_network(self):
        """
        Updates the current TargetNet with the TrainNet which brings all the
        training in the TrainNet to the TargetNet.

        :return: None
        """
        self.TargetNet.set_weights(self.TrainNet.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

class ReplayBuffer:
    def __init__(self):
        self.experiences = deque(maxlen=10_000)
        self.batch_size = 32

    def add_experience(self, exp):
        """
        Stores a step of gameplay experience in
        the buffer for later training
        """
        self.experiences.append((exp['s'], exp['s2'], exp['a'], exp['r'], exp['d']))

    def sample_batch(self):
        """
        Samples a batch of gameplay experiences
        for training purposes.
        """
        batch_size = min(self.batch_size, len(self.experiences))
        sampled_batch = random.sample(self.experiences, batch_size)
        s_batch, ns_batch, a_batch, r_batch, d_batch = [], [], [], [], []
        for exp in sampled_batch:
            s_batch.append(exp[0])
            ns_batch.append(exp[1])
            a_batch.append(exp[2])
            r_batch.append(exp[3])
            d_batch.append(exp[4])

        return np.array(s_batch), np.array(ns_batch), a_batch, r_batch, d_batch