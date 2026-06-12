import numpy as np
from perform_gaussian_mutation import *
from perform_polynomial_mutation import *
from generate_random_chromosome import *

def mutation(op, gas, chromosome):
    """
    Perform mutation on a chromosome based on configured method and mutation probability.
    Supports 'gaussian', 'poly', and 'random' mutation methods.
    """
    if np.random.rand() <= gas['mutation_probability']:
        method = gas['mutation_method']
        if method == 'gaussian':
            return perform_gaussian_mutation(op, gas, chromosome)
        elif method == 'poly':
            return perform_polynomial_mutation(op, gas, chromosome)
        elif method == 'random':
            return generate_random_chromosome(op, gas)
        else:
            raise RuntimeError('Unexpected mutation method.')
    return chromosome
