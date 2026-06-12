import numpy as np
from perform_blx_alpha import *
from perform_blx_alpha_hybrid_avoidance import *

def crossover(op, gas, p1, p2):
    """
    Perform crossover between two parents based on configured method and crossover probability.
    Supports 'blxa' method and hybrid avoidance option.
    """
    if np.random.rand() <= gas['crossover_probability']:
        if gas.get('obstacle_avoidance', False):
            if gas['crossover_method'] == 'blxa':
                return perform_blx_alpha_hybrid_avoidance(op, p1, p2, gas['crossover']['alpha'])
            else:
                raise RuntimeError('Unexpected crossover method for hybrid avoidance.')
        else:
            if gas['crossover_method'] == 'blxa':
                return perform_blx_alpha(op, p1, p2, gas['crossover']['alpha'])
            else:
                raise RuntimeError('Unexpected crossover method.')
    else:
        return p1, p2