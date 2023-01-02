import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam

class ActorModel:
    def __init__(self, n_states, n_actions ,hidden_units):
        super(ActorModel, self).__init__()
        # Input layer
        self.input_layer = InputLayer(input_shape = (n_states,))

        # Hidden layers
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(Dense(i, activation='relu', kernel_initializer='RandomNormal'))

        # Ouput layer
        self.output_layer = Dense(n_actions, activation='softmax')
            
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class CriticalModel:
    def __init__(self, n_states, n_actions, hidden_units):
        # Input layer
        self.input_layer = InputLayer(input_shape = (n_states,))
        self.old_values = InputLayer(input_shape = (1,))

        # Hidden layers
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(Dense(i, activation='relu', kernel_initializer='he_uniform'))

        # Ouput layer
        self.output_layer = Dense(1, activation=None)

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class PPOAgent:
    def __init__(self, n_states, n_actions, hidden_units):
        # Initialise parameters
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 
        self.optimizer = Adam




