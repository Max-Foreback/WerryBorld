import random 
import copy

def tourny(population, tournament_size = 2):
    elite = max(population, key=lambda swarm: population[swarm])
    new_population = [elite]
    for _ in range(len(population) - 1):
        tournament = random.sample(population.keys(), tournament_size)
        winner = max(tournament, key=lambda swarm: population[swarm])  
        new_population.append(copy.deepcopy(winner))

    return new_population