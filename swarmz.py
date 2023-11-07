#fortnite
import random, csv, copy
import matplotlib.pyplot as plt
import pandas as pd
from brainz import simpleNN
from selectorz import tourny
from configz import CONFIG

#random.seed(5)

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


class Swarm():
    #Create new world by default
    def __init__(self, num_agents, world=World(), homogeneous=CONFIG["homogeneous"]):
        self.score = 0
        self.agents = []
        self.world = world

        #brain = simpleNN()
        brain = simpleNN(1)


        for _ in range(num_agents):
            agent_x, agent_y = self.randomize_pos()
            perception = self.world.get_agent_perception(agent_x, agent_y)
            agent = Agent(agent_x, agent_y, perception, brain) if homogeneous else Agent(agent_x, agent_y, perception)
            self.agents.append(agent)
            self.world.set_cell(agent_x, agent_y, 'A')  # Mark the cell as occupied by an agent
    
    def randomize_pos(self):
        agent_x, agent_y = random.randint(0, self.world.xSize - 1), random.randint(0, self.world.ySize - 1)
        while self.world.is_occupied(agent_x, agent_y):
            agent_x, agent_y = random.randint(0, self.world.xSize - 1), random.randint(0, self.world.ySize - 1)
        return agent_x, agent_y
        
    def make_moves(self):
        moves = []
        for agent in self.agents:
            desired_move = agent.calc_move()
            if 0 <= desired_move[0] < len(self.world.grid) and 0 <= desired_move[1] < len(self.world.grid[0]):
                if(desired_move not in moves) and not self.world.is_occupied(desired_move[0], desired_move[1]):
                    moves.append(desired_move)
                    self.world.set_cell(agent.x, agent.y, random.randint(0, 1))
                    agent.x = desired_move[0]
                    agent.y = desired_move[1]
                    agent.score = agent.score + self.world.get_cell(agent.x, agent.y)
                    self.world.set_cell(agent.x, agent.y, 'A')
            else:
                moves.append((agent.x, agent.y))
        for agent in self.agents:
            agent.perception = self.world.get_agent_perception(agent.x, agent.y)
        self.score = sum(agent.score for agent in self.agents)
    
    def reset(self):
        self.score = 0
        self.world = World()
        for agent in self.agents:
            agent.x, agent.y = self.randomize_pos()
            agent.score = 0
            self.world.set_cell(agent.x, agent.y, 'A')


#Should have getters and setters eventually
class Agent():
    #If brain is not passed, swarm is heterogeneous 
    def __init__(self, x, y, perception, brain=simpleNN(1)):
        self.x = x
        self.y = y
        self.perception = perception
        self.score = 0
        self.brain = brain

    def calc_move(self):
        input = [-2 if item == 'A' else item for item in self.perception]
        #output = self.brain.eval(input)
        #output = self.brain.eval(simplify(input))
        output = simplify(input)[0]
        if output > 3: 
            output += 1
        x_offset = (output % 3) - 1
        y_offset = (output // 3) - 1
        return (self.x + x_offset, self.y + y_offset)

def Setup(pop_size=CONFIG["num_swarms"], num_agents=CONFIG["swarm_size"]):
    pop = []
    for _ in range(pop_size):
        pop.append(Swarm(num_agents))
    return pop

def Evaluate(pop, num_timesteps=CONFIG["num_eval_timesteps"]):
    for swarm in pop:
        for _ in range(num_timesteps):
            swarm.make_moves()
    return pop

def Evolve(pop):
    #Selection
    new_pop = tourny(pop)
    #Hacky homogeneous mutation
    for swarm in new_pop:
        temp = swarm.agents[0].brain
        temp.mutate()
        for agent in swarm.agents:
            agent.brain = temp
    return new_pop

def Run(n=CONFIG["num_generations"]):
    pop = Setup()
    tracker = []

    for i in range(n):
        evaluated_pop = Evaluate(pop)
        tracker.append([i, max(evaluated_pop, key=lambda swarm: swarm.score).score])
        evolved_pop = Evolve(evaluated_pop)
        #Dont reset on last iter
        if(i != n-1):
            for swarm in evolved_pop:
                swarm.reset()
        pop = evolved_pop

    return pop, tracker

def display(pop):
    #Final eval for final controllers
    evaluated = Evaluate(pop)
    print(max(evaluated, key=lambda swarm: swarm.score).score)
    # for swarm in evaluated:
    #     print("swarm fitness: ", swarm.score)
    #     #swarm.world.p()
    #     for agent in swarm.agents:
    #         print("agent fitness: ", agent.score)
    #     print("")

def plot(data):
    df = pd.read_csv('out.csv')
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    plt.plot(x, y)
    plt.xlabel('Generations')
    plt.ylabel('Best Swarm Fitness')
    plt.title('Fitness over time')
    plt.show()

def simplify(lst):
    if 2 in lst:
        return [lst.index(2)]
    elif 1 in lst:
        return [lst.index(1)]
    elif 0 in lst:
        return [lst.index(0)]
    else: return [1]

def observe(swarm, n=CONFIG["num_observe"]):
    worlds = []
    worlds.append(swarm.world.get_curr_grid())
    for _ in range(n):
        swarm.make_moves()
        worlds.append(swarm.world.get_curr_grid())
    
    int_worlds = [[[4 if isinstance(item, str) else item for item in row] for row in world] for world in worlds]
    for i, grid in enumerate(int_worlds):
        plt.imshow(grid)
        plt.colorbar()
        plt.title("Update " + str(i) + " score: " + str(swarm.score))
        plt.draw()
        plt.pause(1)
        plt.clf() 
    
final_pop, data = Run()
with open("out.csv", 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(data)

s = max(final_pop, key=lambda swarm: swarm.score)
print(s.score)
observe(max(final_pop, key=lambda swarm: swarm.score))
plot(data)
