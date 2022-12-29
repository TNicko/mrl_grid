import gym
import gym.spaces
import numpy as np
from mrl_grid.window import Window

FPS = 10

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
            low=np.zeros(self.grid_size * 2),
            high=np.ones(self.grid_size * 2),
            dtype=np.float32
        )

        # Rendering
        self.window = None
        self.fps = FPS
    
    def reward_function(self, pos: tuple, action=None) -> int:
        x, y = pos
        if self.grid[x, y] == 0:
            # if np.all(self.grid == 1):
            #     return 100
            return 1
        else:
            return -1

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

        # Make sure coords are in range of grid provided
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        self.current_pos = (x, y)

        # Check if goal has been reached (all tiles visited)
        if np.all(self.grid == 1):
            done = True  # Set the done flag to True
        else:
            done = False

        reward = self.reward_function(self.current_pos, action)
        state = self.pos_to_state(self.current_pos)

        # Grid cell now traversed
        self.grid[x, y] = 1

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
        # State for non-traversed cell
        # if self.grid[x, y] == 0:
        #     state = y * self.width + x
        
        # # State for traversed cell
        # if self.grid[x, y] == 1:
        #     state = (y * self.width + x) + self.grid_size
        
        return state
    
    def state_to_pos(self, state: int) -> tuple:
        """
        Get correct position of state
        """
        if state >= self.grid_size:
            tmp = state - self.grid_size
        else:
            tmp = state

        y = tmp // self.width
        x = tmp % self.width

        return (x, y)
        




