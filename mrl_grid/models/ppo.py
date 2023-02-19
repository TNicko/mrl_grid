import tensorflow as tf 
from tensorflow import keras
from keras.layers import Dense, InputLayer, Flatten
from keras.optimizers import Adam
import numpy as np
import tensorflow_probability as tfp

class Critic(tf.keras.Model):
    """Takes the current state as input and returns the value of the state."""
    def __init__(self):
        super().__init__(self)
        self.d1 = Dense(128, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, inputs):
        x = self.d1(inputs)
        v = self.v(x)
        return v
    
class Actor(tf.keras.Model):
    """Takes the current state as input and returns and outputs the probability of each action."""
    def __init__(self):
        super().__init__(self)
        self.d0 = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.a = Dense(4, activation='softmax')

    def call(self, inputs):
        x = self.d0(inputs)
        z = self.d1(x)
        a = self.a(z)
        return a

class Agent():
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = Adam(learning_rate=0.001)
        self.critic_optimizer = Adam(learning_rate=0.005)
        self.clip_param = 0.2 # used in the actor loss function

    def get_action(self, state):
        """Returns an action given a state."""
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob)
        action = dist.sample()
        return int(action.numpy()[0])

    def learn(self, states, actions, adv, old_probs, discnt_rewards):
        """
        Calculate current probabilities and losses
        Updates the actor and critic networks using the policy gradient method.
        """

        discnt_rewards = tf.reshape(discnt_rewards, [-1])
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        adv = tf.reshape(adv, [-1])
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs
        old_p = tf.reshape(old_p, (len(old_p), 4))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            print("states shape in tape: ", states.shape)
            print("V shape: ", v.shape)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * keras.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
        
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return a_loss, c_loss

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(entropy)
        # print(probability)
        sur1 = []
        sur2 = []

        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            t = tf.constant(t)
            ratio = tf.math.divide(pb[a], op[a])
            s1 = tf.math.multiply(ratio, t)
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0-self.clip_param, 1.0+self.clip_param), t)
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)

        return loss
    


# class ActorModel:
#     def __init__(self, n_states, n_actions ,hidden_units):
#         super(ActorModel, self).__init__()
#         # Input layer
#         self.input_layer = InputLayer(input_shape = (n_states,))

#         # Hidden layers
#         self.hidden_layers = []
#         for i in hidden_units:
#             self.hidden_layers.append(Dense(i, activation='relu', kernel_initializer='RandomNormal'))

#         # Ouput layer
#         self.output_layer = Dense(n_actions, activation='softmax')
            
#     @tf.function
#     def call(self, inputs):
#         z = self.input_layer(inputs)
#         for layer in self.hidden_layers:
#             z = layer(z)
#         output = self.output_layer(z)
#         return output


# class CriticalModel:
#     def __init__(self, n_states, n_actions, hidden_units):
#         # Input layer
#         self.input_layer = InputLayer(input_shape = (n_states,))
#         self.old_values = InputLayer(input_shape = (1,))

#         # Hidden layers
#         self.hidden_layers = []
#         for i in hidden_units:
#             self.hidden_layers.append(Dense(i, activation='relu', kernel_initializer='he_uniform'))

#         # Ouput layer
#         self.output_layer = Dense(1, activation=None)

#     @tf.function
#     def call(self, inputs):
#         z = self.input_layer(inputs)
#         for layer in self.hidden_layers:
#             z = layer(z)
#         output = self.output_layer(z)
#         return output


# class PPOAgent:
#     def __init__(self, n_states, n_actions, hidden_units):
#         # Initialise parameters
#         self.max_average = 0 # when average score is above 0 model will be saved
#         self.lr = 0.00025
#         self.epochs = 10 
#         self.optimizer = Adam




