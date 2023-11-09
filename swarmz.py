import random, copy
from brainz import simpleNN
from configz import CONFIG
from worldz import World
import torch

class Swarm():
    def __init__(self, num_agents=CONFIG["swarm_size"], homogeneous=CONFIG["homogeneous"]):
        self.score = 0
        self.agents = []

        #brain = simpleNN()
        brain = simpleNN()

        for _ in range(num_agents):
            agent = Agent(brain) if homogeneous else Agent()
            self.agents.append(agent)

    def spawn_swarm(self, world):
        for agent in self.agents:
            agent_x, agent_y = random.randint(0, world.xSize - 1), random.randint(0, world.ySize - 1)
            while world.is_occupied(agent_x, agent_y):
                agent_x, agent_y = random.randint(0, world.xSize - 1), random.randint(0, world.ySize - 1)
            world.set_occupied(agent_x, agent_y)
            agent.set_position(agent_x, agent_y)

    def make_moves(self, world):
        scored = 0
        for agent in self.agents:
            agent.set_perception(world.get_agent_perception(agent.position))
            desired_move = agent.calc_move()
            new_x = desired_move[0]
            new_y = desired_move[1]
            #if move is valid
            if world.is_pos_valid(new_x, new_y) and not world.is_occupied(new_x, new_y):
                #Replace old location with new val, update agent pos, update score, update new location, update agent perception
                world.set_random(agent.get_x(), agent.get_y())
                agent.position = desired_move
                scored += world.get_cell(new_x, new_y)
                world.set_occupied(new_x, new_y)
                agent.set_perception(world.get_agent_perception(agent.position))
        return scored

class Agent():
    #If brain is not passed, swarm is heterogeneous 
    def __init__(self, brain=None):
        if brain is None:
            brain = simpleNN()
        self.position = (None, None)
        self.perception = {}
        self.brain = brain

    def set_position(self, x, y):
        self.position = (x, y)

    def set_perception(self, p):
        self.perception = p

    def get_position(self):
        return self.position

    def get_x(self):
        return self.position[0]

    def get_y(self):
        return self.position[1]

    def calc_move(self):
        #output = self.brain.eval(input)
        #output = self.brain.eval(simplify(input))
        #output = simplify(self.perception)[0]
        #Hacky make 4 work
        # if output > 3: 
        #     output += 1
        # x_offset = (output % 3) - 1
        # y_offset = (output // 3) - 1
        out = self.brain.get_outputs(torch.tensor(list(self.perception.values()), dtype=torch.float32))
        x_out = out[0].item()
        y_out = out[1].item()
        #temp
        x_offset = -1 if x_out < -0.3 else (0 if -0.3 <= x_out <= 0.3 else 1)
        y_offset = -1 if y_out < -0.3 else (0 if -0.3 <= y_out <= 0.3 else 1)
        #Greedy
        # key_with_highest_value = max(self.perception, key=lambda k: self.perception[k])
        # return key_with_highest_value
        return (self.position[0] + x_offset, self.position[1] + y_offset)
    
def simplify(perception):
    return max(perception, key=lambda k: perception[k])
    if 2 in lst:
        return [lst.index(2)]
    elif 1 in lst:
        return [lst.index(1)]
    elif 0 in lst:
        return [lst.index(0)]
    else: return [1]
