import random 
import copy

def tourny(population, tournament_size = 2):
    new_population = []

    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda swarm: swarm.score)  
        new_population.append(copy.deepcopy(winner))

    return new_population