from calculate_static_penalty import *

def check_constraints(op, gas, population, fit_array):
    """
    Apply constraint penalties to the fitness array based on static penalty method.
    Modifies fitness penalties and adds penalties to IK fitness.
    """
    if gas['penalty_method'] == 'static':
        for i in range(gas['n_individuals']):
            chromosome_id = int(fit_array[i, gas['fitIdx']['id']])
            chromosome = population[:, :, chromosome_id]
            penalty = calculate_static_penalty(op, gas, chromosome)
            fit_array[i, gas['fitIdx']['penalty']] = penalty
            fit_array[i, gas['fitIdx']['ikFitness']] += penalty
    else:
        raise RuntimeError("Unexpected or unimplemented constraint handling method.")
    return fit_array
