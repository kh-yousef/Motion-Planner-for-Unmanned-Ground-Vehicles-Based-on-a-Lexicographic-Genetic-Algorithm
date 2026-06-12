import numpy as np
from calculate_fitness import *

def evaluate(op, gas, population):
    """
    Evaluate population fitness metrics.
    Returns updated population and fitness array.
    """
    n_individuals = gas['n_individuals']
    n_metrics = len(gas['fitIdx'])
    fit_array = np.zeros((n_individuals, n_metrics))
    evaluated_pop = np.copy(population)

    for i in range(n_individuals):
        chromosome = population[:, :, i]
        mod_chrom, ik_fitness, node_count, undulation, path_len, orn_error = calculate_fitness(op, gas, chromosome)

        evaluated_pop[:, :, i] = mod_chrom

        fit_array[i, gas['fitIdx']['ikFitness']] = ik_fitness
        fit_array[i, gas['fitIdx']['ikFitnessModified']] = np.floor(ik_fitness / gas['ranking']['step_ik']) * gas['ranking']['step_ik']
        fit_array[i, gas['fitIdx']['pathLengthModified']] = np.floor(path_len / gas['ranking']['step_path_len']) * gas['ranking']['step_path_len']
        fit_array[i, gas['fitIdx']['nodeCount']] = node_count
        fit_array[i, gas['fitIdx']['undulation']] = undulation
        fit_array[i, gas['fitIdx']['pathLength']] = path_len
        fit_array[i, gas['fitIdx']['ornError']] = orn_error
        fit_array[i, gas['fitIdx']['penalty']] = 0
        fit_array[i, gas['fitIdx']['rank']] = 0
        fit_array[i, gas['fitIdx']['id']] = i

    return evaluated_pop, fit_array