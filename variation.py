import numpy as np
from crossover import *
from mutation import *

def variation(op, gas, pop, mat_pool):
    """
    Generate offspring population via crossover and mutation on the selected mating pool.
    Pairs are selected in shuffled order from mat_pool.
    """
    n_individuals = gas['n_individuals']
    offspring = np.zeros_like(pop)

    shuffled_pool = np.random.permutation(mat_pool)

    for i in range(0, n_individuals, 2):
        parent1 = pop[:, :, shuffled_pool[i]]
        parent2 = pop[:, :, shuffled_pool[i + 1]]
        child1, child2 = crossover(op, gas, parent1, parent2)
        child1 = mutation(op, gas, child1)
        child2 = mutation(op, gas, child2)
        offspring[:, :, i] = child1
        offspring[:, :, i + 1] = child2

    return offspring