import random


class NoLearning:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, type=None):

        action = self.env.action_space.sample() # Choose random available action

        return action