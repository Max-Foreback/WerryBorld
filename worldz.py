from configz import CONFIG
import random, copy

#INNER ARE COLS
class World:
    def __init__(self, x=CONFIG["world_x"], y=CONFIG["world_y"]):
        self.grid = [[random.random() > .75 for _ in range(x)] for _ in range(y)]
        self.xSize = x
        self.ySize = y
        self.total_score = 0

    def get_cell(self, x, y):
        return self.grid[x][y]
    
    def get_grid(self):
        return self.grid
    
    def get_curr_grid(self):
        return copy.deepcopy(self.grid)

    def set_cell(self, x, y, value):
        self.grid[x][y] = value

    def is_occupied(self, x, y):
        return self.grid[x][y] == 'A'
    
    def get_agent_perception(self, agent_x, agent_y):
        perception = []
        for i in range(agent_x - 1, agent_x + 2):
            for j in range(agent_y - 1, agent_y + 2):
                #Dont need to percieve agents own pos
                if (i == agent_x and j == agent_y):
                    pass
                #Detect agents surroundings, reading style (left to right)
                elif 0 <= i < len(self.grid) and 0 <= j < len(self.grid[0]):
                    perception.append(self.grid[i][j])
                else:
                    perception.append(-1)
        return perception