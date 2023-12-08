import csv, copy, random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from selectorz import tourny
from configz import CONFIG
from swarmz import Swarm
from worldz import World

#random.seed(4)

def Setup(pop_size=CONFIG["num_swarms"], num_agents=CONFIG["swarm_size"]):
    pop = []
    for _ in range(pop_size):
        pop.append(Swarm(num_agents))
    return pop

def eval_pop(pop):
    evaluated = {}
    for swarm in pop:
        score = 0
        evals = 5
        for _ in range(evals):
            w = World()
            p = w.get_empty_positions(CONFIG["swarm_size"])
            score += eval_swarm(swarm, world=copy.deepcopy(w), initial_positions=copy.deepcopy(p))
        avg_score = score/evals
        evaluated[swarm] = avg_score
    return evaluated

def eval_swarm(swarm, world=None, num_timesteps=CONFIG["num_eval_timesteps"], track=False, initial_positions=None):
    if world is None:
        world = World()
    if initial_positions is None:
        initial_positions = world.get_empty_positions(CONFIG["swarm_size"])
    swarm_score = 0
    swarm.spawn_swarm(world, positions=initial_positions)
    # If we're tracking save data for the vizualization 
    if track:
        snapshots = []
        scores = [0]
        snapshots.append(world.get_grid_cpy())
    for _ in range(num_timesteps):
        swarm_score += swarm.make_moves(world)
        if track:
            snapshots.append(world.get_grid_cpy())
            scores.append(swarm_score)
    if track: 
        return snapshots, scores
    return swarm_score  

def Evolve(eval_pop, h=CONFIG["homogeneous"]):
    #Selection
    new_pop = tourny(eval_pop)
    #Hacky homogeneous mutation (maybe not even needed if refs are shared?)
    if h:
        for swarm in new_pop:
            temp = swarm.agents[0].brain
            temp.mutate()
            for agent in swarm.agents:
                agent.brain = temp
    else:
        for swarm in new_pop:
            for agent in swarm.agents:
                agent.brain.mutate()
    return new_pop

def runner(n=CONFIG["num_generations"]):
    pop = Setup()
    tracker = []

    for i in range(n):
        #Need to flush buffer for HPCC printing
        print("Generation " + str(i), flush=True)
        evaluated_pop = eval_pop(pop)
        avg_fitness = sum(evaluated_pop.values())/len(evaluated_pop)
        max_fitness = max(evaluated_pop.values())
        tracker.append([i, max_fitness, avg_fitness])
        evolved_pop = Evolve(evaluated_pop)
        pop = evolved_pop

    return pop, tracker

def greedy():
    n = CONFIG["num_generations"]
    swarm_size = CONFIG["swarm_size"]
    evals = CONFIG["num_eval_timesteps"]
    greedy_tracker = []

    for _ in range(n):
        world = World()
        swarm = Swarm(swarm_size)
        swarm_score = 0
        swarm.spawn_swarm(world)
        for _ in range(evals):
            swarm_score += swarm.make_moves(world, greedy=True)
        greedy_tracker.append(swarm_score)
    return greedy_tracker


def plot():
    df = pd.read_csv('results/out.csv')
    gen = df.iloc[:, 0]
    max_fitness = df.iloc[:, 1]
    avg_fitness = df.iloc[:, 2]
    greedy_fitness = df.iloc[:, 3]
    # clear plot
    plt.clf()
    plt.figure(figsize=(40, 10), dpi=150)
    plt.plot(gen, max_fitness, label='Max Fitness')
    plt.plot(gen, avg_fitness, label='Avg Fitness')
    plt.plot(gen, greedy_fitness, label='Greedy Fitness')
    plt.axhline(y=25, color='green')
    plt.xlabel('Generations')
    plt.ylabel('Best Swarm Fitness')
    plt.title('Fitness over time')
    plt.legend()
    plt.savefig('results/fitness.png')

def observe(swarm, n=CONFIG["num_observe"]):
    snapshots, scores = eval_swarm(swarm, num_timesteps=n, track=True)
    #rotate for proper printing
    rotated = []
    for snapshot in snapshots:
        rows = list(map(list, zip(*snapshot)))
        rotated.append(rows)

    def update(frame):
        plt.clf() 
        plt.imshow(rotated[frame], origin='lower')
        plt.colorbar()
        plt.title("Update " + str(frame) + " Score: " + str(scores[frame]))

    try:
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=len(rotated), interval=200)
        #make sure you have ffmpeg installed 
        ani.save('results/output.mp4', writer='ffmpeg', fps=1)
    except Exception as e:
        #if you cant get ffmpeg to work, this is a valid alternative 
        for i in range(len(rotated)):
            plt.imshow(rotated[i], origin='lower')
            plt.colorbar()
            plt.title("Update " + str(i) + " Score: " + str(scores[i]))
            plt.savefig(f"results/plot_{i}.png")
            plt.clf()  
        print("Unable to generate mp4 file. Error: ", str(e))

if __name__ == "__main__":

    final_pop, data = runner()
    greedy_data = greedy()
    for i in range(len(data)):
        data[i].append(greedy_data[i])

    with open("results/out.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)
    eval_final_pop = eval_pop(final_pop)
    #Observe elite
    observe(max(eval_final_pop, key=lambda swarm: eval_final_pop[swarm]))
    plot()
