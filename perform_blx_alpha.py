import numpy as np
from blend_values import *
def perform_blx_alpha(op, p1, p2, alpha):
    """
    Perform BLX-alpha crossover for two parents (without obstacle-aware adjustment).
    Blend lengths and angles separately using alpha parameter.
    """
    n_targets = p1.shape[0] - 1
    o1 = np.zeros_like(p1)
    o2 = np.zeros_like(p2)

    for j in range(op['n_nodes']):
        o1[-1, j], o2[-1, j] = blend_values(p1[-1, j], p2[-1, j], alpha, op['length_domain'])

    for i in range(n_targets):
        for j in range(op['n_nodes']):
            o1[i, j], o2[i, j] = blend_values(p1[i, j], p2[i, j], alpha, op['angle_domain'])

    return o1, o2