import numpy as np
from generate_random_chromosome import *

def initialize_random_population(op, gas):
    """
    Initialize a population for the genetic algorithm.
    Population shape: (n_targets+1, n_nodes + extra_genes, n_individuals).
    Each individual is a chromosome representing path parameters.
    """
    n_targets = op['targets'].shape[0]
    pop = np.zeros((n_targets + 1, op['n_nodes'] + gas['extra_genes'], gas['n_individuals']))

    for i in range(gas['n_individuals']):
        pop[:, :, i] = generate_random_chromosome(op, gas)

    return pop