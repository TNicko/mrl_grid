import gym
import gym.spaces
import numpy as np
from mrl_grid.window import Window
from mrl_grid.world import World, Agent, SeenCell, AGENT_COLORS

FPS = 20

REWARD_MAP = {
    'illegal': -0.5,
    'new': 1,
    'move': -0.05,
    'wait': -0.1,
    'collision': -10,
    'goal': 100,
}

class MultiGridEnv(gym.Env):
    def __init__(self, grid_map, n_channels):
            
        self.rows, self.cols = self.grid_map.shape
        self.grid_size = self.cols * self.rows
        self.channels = n_channels

        self.grid = np.asarray(grid_map)
        self.world = self._initialise_world()
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(self.agents)

        self.shared_reward = True

        self.nA = 5 # no of actions (up, down, left, right, wait)
        self.action_space = gym.spaces.Discrete(self.nA)

        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(
                np.zeros((self.rows, self.cols, self.channels)),    
                np.ones((self.rows, self.cols, self.channels)),
            ) for _ in self.agents
        ])

        # Rendering
        self.window = None
        self.fps = FPS

    def _initialise_world(self):
        world = World()

        num_agents = 0
        for row in range(self.rows):
            for col in range(self.cols):
                # Initialise agents in grid
                if self.grid[row, col] == 1:
                    agent = Agent()
                    agent.name = 'agent %d' % num_agents
                    agent.collide = True
                    agent.color = AGENT_COLORS[num_agents]['color']
                    agent.color_trail = AGENT_COLORS[num_agents]['color_trail']
                    agent.state.pos = (row, col)
                    world.agents.append(agent)
                    num_agents += 1

                    # mark cell on grid as visited
                    world.cell_visited(agent)

        return world
            
    def step(self, action_n):
        assert len(action_n) == len(self.agents)

        # set action for each agent
        reward_n = []
        info_n = []
        for i, agent in enumerate(self.agents):
            agent_pos = agent.state.pos
            new_pos = agent.get_new_pos(action_n[i])
            reward, updated_pos = self._get_reward(agent, new_pos)
            reward_n.append(reward)
            agent.state.pos = updated_pos
            
            info_n.append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        # Check if goal has been reached (all tiles visited)
        done = False
        if np.all(self.grid[:,:,1] == 1):
            done = True  # Set the done flag to True

        # Compute next observations for each agent
        state_n = []
        for agent in self.agents:
            state_n.append(self._get_obs(agent))

        return state_n, reward_n, done, info_n

    def _get_obs(self, agent):
        """Get observation for agent"""
        pass

    def _get_reward(self, agent, new_pos):
        x, y = new_pos
        reward = 0
        updated_pos = new_pos

        # Negative rewards: collide against obstacles/other agents,

        # ------------------------------------------------------------

        # Illegal move outside of grid boundary
        if x != max(0, min(x, self.cols - 1)) or y != max(0, min(y, self.rows - 1)):
            reward += REWARD_MAP['illegal']
            updated_pos = agent.state.pos

        # moved to new grid cell
        elif self.grid[x, y, 1] == 0:
            self.grid[x, y, 1] = 1 # Mark grid cell as traversed
            reward += 1
            self.world.cell_visited(agent)
            
            # All of grid explored
            if np.all(self.grid[:,:,1] == 1):
                reward += REWARD_MAP['goal']

        # movement cost
        if new_pos != agent.state.pos:
            reward += REWARD_MAP['move'] 

        # wait cost
        if new_pos == agent.state.pos:
            reward += REWARD_MAP['wait']
        
        return updated_pos, reward

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







class GridEnv(gym.Env):
    def __init__(self, grid_map, n_channels, start_pos):

        self.grid_map = np.asarray(grid_map)
        self.rows, self.cols = self.grid_map.shape
        self.grid_size = self.cols * self.rows
        self.channels = n_channels

        self.start_pos = start_pos
        self.current_pos = start_pos

        self.grid = self.initialise_grid()

        self.nA = 4 # no of actions (up, down, left, right)
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

        info = {}

        return state, reward, done, info

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

    def initialise_grid(self):
        self.current_pos = self.start_pos

        # Create grid array
        grid = np.zeros((self.rows, self.cols, self.channels))

        # Set agent start position on grid
        x, y = self.start_pos
        grid[x, y, 0] = 1
        grid[x, y, 1] = 1

        return grid









