import numpy as np
import tensorflow as tf
import random

class MyModel(tf.keras.Model):
    def __init__(self, observation_space, hidden_units, n_actions):
        super(MyModel, self).__init__()
        # Input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape = observation_space.shape)
        
        # Hidden layers
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(
                tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))

        # Ouput layer
        self.output_layer = tf.keras.layers.Dense(
            n_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        print(output)
        return output



