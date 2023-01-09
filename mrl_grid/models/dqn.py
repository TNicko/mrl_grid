import numpy as np
import tensorflow as tf
from mrl_grid.models.models import MyModel

class DQN:
    def __init__(
        self, 
        observation_space, 
        nA):
        # Initialise parameters
        self.render = False
        self.nA = nA
        self.hidden_units = [10, 10]
        self.lr = 0.1       # learning rate
        self.gamma = 0.9    # discount factor (value between 0,1)
        self.epsilon = 0.1  # decays overtime (value between 0,1)
        self.min_epsilon = 0.1
        self.decay = 0.99
        self.max_experiences = 10_000
        self.min_experiences = 100
        self.batch_size = 32
        self.copy_step = 25
        self.optimizer = tf.optimizers.Adam(self.lr)
        self.model = MyModel(observation_space, self.hidden_units, nA)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}


    def predict(self, inputs):
        return self.model(inputs.astype('float32'))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.nA), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return loss

    def get_action(self, states):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.nA)
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

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)