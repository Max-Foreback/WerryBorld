from configz import CONFIG
import random, copy

#INNER ARE COLS
class World:
    def __init__(self, x=CONFIG["world_x"], y=CONFIG["world_y"]):
        self.grid = [[random.randint(0, 1) for _ in range(x)] for _ in range(y)]
        self.xSize = x
        self.ySize = y
        self.total_score = 0

    def get_cell(self, x, y):
        return copy.deepcopy(self.grid[x][y])
    
    def get_grid(self):
        return self.grid
    
    def get_grid_cpy(self):
        return copy.deepcopy(self.grid)

    def set_cell(self, x, y, value):
        self.grid[x][y] = value

    def set_occupied(self, x, y):
        assert self.grid[x][y] != -2
        self.grid[x][y] = -2

    def set_random(self, x, y):
        self.grid[x][y] = random.randint(0, 1)

    def is_occupied(self, x, y):
        return self.grid[x][y] == -2
    
    def is_pos_valid(self, x, y):
        if 0 <= x < self.xSize and 0 <= y < self.ySize:
            return True
        return False 
    
    def get_agent_perception(self, agent_pos):
        agent_x = agent_pos[0]
        agent_y = agent_pos[1]
        perception = {}
        for i in range(agent_x - 1, agent_x + 2):
            for j in range(agent_y - 1, agent_y + 2):
                #Dont need to percieve agents own pos
                if (i == agent_x and j == agent_y):
                    continue
                #Detect agents surroundings, reading style (left to right)
                elif 0 <= i < len(self.grid) and 0 <= j < len(self.grid[0]):
                    perception[i, j] = self.get_cell(i, j)
                else:
                    perception[i, j] = -1
        return perception