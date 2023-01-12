import gym
import gym.spaces
import numpy as np
from mrl_grid.window import Window

FPS = 20

class GridEnv(gym.Env):
    def __init__(self, cols, rows, n_channels, start_pos):

        self.cols = cols
        self.rows = rows
        self.grid_size = cols * rows
        self.channels = n_channels

        self.start_pos = start_pos
        self.current_pos = start_pos

        self.grid = self.initialise_grid()

        self.nA = 4 # no of actions (up, down, left, right)
        self.nS = 2 ** (self.grid_size * 2) * self.grid_size # no of states
        self.action_space = gym.spaces.Discrete(self.nA)
        
        self.observation_space = gym.spaces.Box(
            np.zeros((self.rows, self.cols, self.channels)),    
            np.ones((self.rows, self.cols, self.channels)),  
        )

        # Rendering
        self.window = None
        self.fps = FPS
    
    def reward_function(self, pos: tuple, action=None) -> int:
        x, y = pos
        reward = 0

        # Illegal move outside of grid boundary
        if x != max(0, min(x, self.cols - 1)) or y != max(0, min(y, self.rows - 1)):
            reward += -0.5

        # moved to new grid cell
        elif self.grid[x, y, 1] == 0:
            self.grid[x, y, 1] = 1 # Mark grid cell as traversed
            reward += 1
            
            # All of grid explored
            if np.all(self.grid[:,:,1] == 1):
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
        x = max(0, min(x, self.cols - 1))
        y = max(0, min(y, self.rows - 1))
        self.current_pos = (x, y)

        # Update agent position on grid
        self.grid[:,:,0] = 0
        self.grid[x, y, 0] = 1

        # Check if goal has been reached (all tiles visited)
        if np.all(self.grid[:,:,1] == 1):
            done = True  # Set the done flag to True
        else:
            done = False

        # state = self.flatten_grid(self.grid)
        state = self.grid

        return state, reward, done

    def reset(self):
        self.grid = self.initialise_grid()
        # start_state = self.flatten_grid(self.grid)
        return self.grid

    def render(self, mode='human'):
        if mode == "human":
            self.render_gui()

    def render_gui(self):
        if self.window == None:
            self.window = Window('Grid World', self.cols, self.rows, self.fps)
            self.window.show()

        self.window.render(self.grid, self.current_pos)

    def close(self):
        if self.window:
            self.window.close()
        return

    # def flatten_grid(self, grid):
    #     """
    #     Flatten grid into a single vector
    #     """
    #     flat_grid = grid.flatten()
    #     grid_array = np.array(flat_grid, dtype=np.float32)
    #     return grid_array

    def initialise_grid(self):
        self.current_pos = self.start_pos

        # Create grid array
        grid = np.zeros((self.rows, self.cols, self.channels))

        # Set agent start position on grid
        x, y = self.start_pos
        grid[x, y, 0] = 1
        grid[x, y, 1] = 1

        return grid









