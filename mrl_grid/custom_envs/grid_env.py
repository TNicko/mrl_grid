import gym
import gym.spaces
import numpy as np
from mrl_grid.window import Window

FPS = 20

class GridEnv(gym.Env):
    def __init__(self, width, height, start_state):

        self.width = width
        self.height = height
        self.grid_size = width * height

        self.grid = np.zeros((self.width, self.height))
        self.initial_state = start_state
        self.current_pos = start_state

        self.nA = 4 # no of actions
        self.action_space = gym.spaces.Discrete(self.nA)  # up, down, left, right

        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.grid_size),
            high=np.ones(self.grid_size),
            dtype=np.float32
        )

        # Rendering
        self.window = None
        self.fps = FPS
    
    def reward_function(self, pos: tuple, action=None) -> int:
        x, y = pos
        reward = 0

        # Illegal move outside of grid boundary
        if x != max(0, min(x, self.width - 1)) or y != max(0, min(y, self.height - 1)):
            reward += -0.5

        # moved to new grid cell
        elif self.grid[x, y] == 0:
            self.grid[x, y] = 1 # Mark grid cell as traversed
            reward += 1
            
            # All of grid explored
            if np.all(self.grid == 1):
                reward += 100
 
        reward += -0.05 # movement cost
        
        return reward
            

    def step(self, action):
        x, y = self.current_pos

        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1

        self.current_pos = (x, y)
        reward = self.reward_function(self.current_pos, action)

        # Make sure coords are in range of grid provided
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        self.current_pos = (x, y)

        # Check if goal has been reached (all tiles visited)
        if np.all(self.grid == 1):
            done = True  # Set the done flag to True
        else:
            done = False

        state = self.pos_to_state(self.current_pos)

        return state, reward, done

    def reset(self):
        self.grid = np.zeros((self.width, self.height)) # reset grid
        start_state = self.pos_to_state(self.initial_state)
        return start_state

    def render(self, mode='human'):
        if mode == "human":
            self.render_gui()

    def render_gui(self):

        if self.window == None:
            self.window = Window('Grid World', self.width, self.height, self.fps)
            self.window.show()

        self.window.render(self.grid, self.current_pos)

    def close(self):
        if self.window:
            self.window.close()
        return

    def pos_to_state(self, pos: tuple) -> int:
        """
        Get correct state index of current position
        """
        x, y = pos
        state = y * self.width + x
        
        return state
    
    def state_to_pos(self, state: int) -> tuple:
        """
        Get correct position of state
        """
        y = state // self.width
        x = state % self.width

        return (x, y)
        




