import numpy as np
import tensorflow as tf
from mrl_grid.models.models import MyModel

class DQN:
    def __init__(
        self, 
        n_states, 
        n_actions, 
        hidden_units, 
        gamma, 
        max_experiences, 
        min_experiences, 
        batch_size, 
        lr):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(n_states, hidden_units, n_actions)
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
                self.predict(states) * tf.one_hot(actions, self.n_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

            return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.n_actions)
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