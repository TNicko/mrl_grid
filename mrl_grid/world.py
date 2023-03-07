import numpy as np

AGENT_COLORS = [
    {
    'color': '#e8074a',
    'color_trail': '#e68aa5',
    },
    {
    'color': '#003366',
    'color_trail': '#6da4db',
    },
    {
    'color': '#FF6600',
    'color_trail': '#faa66e',
    },
]

class EntityState(object):
    def __init__(self):
        self.pos = None

class Entity(object):
    def __init__(self):
        self.name = ''
        self.size = 0.050
        self.movable = False
        self.collide = True
        self.color = None
        self.state = EntityState()

class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.movable = True
        self.action = None
        self.size = 0.15
        self.color_trail = None

    def get_new_pos(self, action):
        x, y = self.state.pos

        if action == 0: x -= 1  # up
        if action == 1: x += 1  # down
        if action == 2: y -= 1  # left
        if action == 3: y += 1  # right
        if action == 4: pass    # wait

        return (x, y)


class SeenCell(Entity):
    def __init__(self):
        self.collide = False

# multi-agent world
class World(object):
    def __init__(self):
        self.agents = []
        self.seenCells = []
        
    @property
    def entities(self):
        "return all entities"
        return self.agents
    
    @property
    def agents(self):
        "return all agents"
        return [agent for agent in self.agents]

    def cell_visited(self, agent):
        "mark new cell as visited"
        cell = SeenCell()
        cell.state.pos = agent.state.pos
        cell.color = agent.color_trail
        self.seenCells.append(cell)
