import numpy as np
from ranking_evaluation import *
def elitist_survival(gas, pop, offspring, fit_array_P, fit_array_O):
    """
    Combine parent and offspring populations and select the best individuals.
    Ranks combined fitness and selects top n_individuals as survivors.
    """
    combined_pop = np.concatenate((pop, offspring), axis=2)
    fit_array_O[:, gas['fitIdx']['id']] += gas['n_individuals']  # Adjust IDs in offspring

    combined_fit_array = np.vstack((fit_array_P, fit_array_O))
    ranked_combined_fit = ranking_evaluation(gas, combined_fit_array)

    survivor_fit_data = ranked_combined_fit[:gas['n_individuals'], :]
    next_pop = np.zeros((pop.shape[0], pop.shape[1], gas['n_individuals'])) # Use shape from gas

    for i in range(gas['n_individuals']):
        survivor_id = int(survivor_fit_data[i, gas['fitIdx']['id']])
        next_pop[:, :, i] = combined_pop[:, :, survivor_id]

    next_fit_array = survivor_fit_data
    next_fit_array[:, gas['fitIdx']['id']] = np.arange(gas['n_individuals'])

    return next_pop, next_fit_array