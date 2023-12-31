from configz import CONFIG, resources, probs
import random, copy

#INNER ARE COLS
class World:
    def __init__(self, x=CONFIG["world_x"], y=CONFIG["world_y"]):
        self.grid = [[0 for _ in range(y)] for _ in range(x)]
        self.xSize = x
        self.ySize = y
        self.total_score = 0
        for _ in range(25):
            x, y = random.randint(0, self.xSize - 1), random.randint(0, self.ySize - 1)
            while self.get_cell(x, y) == 1:  
                x, y = random.randint(0, self.xSize - 1), random.randint(0, self.ySize - 1)
            self.set_cell(x, y, 1)

    def get_cell(self, x, y):
        return copy.deepcopy(self.grid[x][y])
    
    def get_cell_score(self, x, y):
        return int(self.grid[x][y])
    
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
        self.grid[x][y] = random.choices(resources, probs)[0]

    def is_occupied(self, x, y):
        if not self.is_pos_valid(x, y):
            return False
        return self.grid[x][y] == -2
    
    def is_pos_valid(self, x, y):
        if 0 <= x < self.xSize and 0 <= y < self.ySize:
            return True
        return False 
    
    def get_empty_positions(self, num_positions):
        positions = []
        while len(positions) < num_positions:
            x, y = random.randint(0, self.xSize - 1), random.randint(0, self.ySize - 1)
            if (x, y) not in positions and self.get_cell(x, y) == 0:
                positions.append((x, y))
        return positions
    
    def get_empty_position(self):
        x, y = random.randint(0, self.xSize - 1), random.randint(0, self.ySize - 1)
        while self.get_cell(x, y) != 0:
            x, y = random.randint(0, self.xSize - 1), random.randint(0, self.ySize - 1)
        return(x, y)

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
                #Lets be really clear about what agents see
                if not self.is_pos_valid(i, j):
                    perception[i, j] = -2
                elif self.is_occupied(i, j):
                    perception[i, j] = -1
                elif self.get_cell(i, j) == 0:
                    perception[i, j] = 0
                elif self.get_cell(i, j) == 1:
                    perception[i, j] = 1
                else:
                    raise Exception("Uh oh")
        return perception