import random


class NoLearning:
    def __init__(self, env):
        self.env = env

    def select_action(self, state, type=None):

        action = self.env.action_space.sample() # Choose random available action

        # Get action based on whether next state's grid cell is new or old
        if type == "assist":
            possible_next_states = self.get_possible_next_states(state)
            for a, s in list(possible_next_states.items()):
                pos = self.env.state_to_pos(s)
                if self.env.grid[pos[0], pos[1]] == 1:
                    possible_next_states.pop(a)

            if len(possible_next_states) != 0:
                action = random.choice(list(possible_next_states))

        return action

    def get_possible_next_states(self, state):
        """
        Get list of possible next states from current state 
        """
        states = dict()

        if state % self.env.rows != 0 and state > 0:
            states[0] = state - 1 # Up
        
        if state % self.env.rows != self.env.rows - 1:
            states[1] = state + 1 # Down

        if state >= self.env.rows:
            states[2] = state - self.env.rows # Left
            
        if state < self.env.grid_size - self.env.rows:
            states[3] = state + self.env.rows # Right

        return states