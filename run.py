import csv, copy
import matplotlib.pyplot as plt
import pandas as pd
from selectorz import tourny
from configz import CONFIG
from swarmz import Swarm

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
