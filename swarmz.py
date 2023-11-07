#fortnite
import random, copy
from brainz import simpleNN
from configz import CONFIG

#random.seed(5)

class Swarm():
    #Create new world by default
    def __init__(self, num_agents, homogeneous=CONFIG["homogeneous"]):
        self.score = 0
        self.agents = None
        self.agent_positions = None

        #brain = simpleNN()
        brain = simpleNN(1)

        for _ in range(num_agents):
            agent_x, agent_y = self.randomize_pos()
            agent = Agent(agent_x, agent_y, brain) if homogeneous else Agent(agent_x, agent_y)
            self.agents.append(agent)
            self.agent_positions.append((agent_x, agent_y))

    def spawn_swarm(self):
        pass
    
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
    def __init__(self, brain=simpleNN(1)):
        self.pos = (None, None)
        self.perception = None
        self.score = 0
        self.brain = brain

    def calc_move(self):
        input = [-2 if item == 'A' else item for item in self.perception]
        #output = self.brain.eval(input)
        #output = self.brain.eval(simplify(input))
        output = simplify(input)[0]
        #Hacky make 4 work
        if output > 3: 
            output += 1
        x_offset = (output % 3) - 1
        y_offset = (output // 3) - 1
        return (self.pos[0] + x_offset, self.pos[1] + y_offset)
    
def simplify(lst):
    if 2 in lst:
        return [lst.index(2)]
    elif 1 in lst:
        return [lst.index(1)]
    elif 0 in lst:
        return [lst.index(0)]
    else: return [1]
