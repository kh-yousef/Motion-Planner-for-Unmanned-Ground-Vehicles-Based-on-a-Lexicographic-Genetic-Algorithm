import numpy as np
def perform_polynomial_mutation(op, gas, chromosome):
    """
    Polynomial mutation applied gene-wise based on mutation and gene-swap probabilities.
    """
    n_targets = chromosome.shape[0] - 1
    eta = gas['mutation']['eta']

    for i in range(n_targets + 1):
        for j in range(op['n_nodes']):
            if np.random.rand() < gas['mutation']['gene_swap_probability']:
                y = chromosome[i, j]
                if i < n_targets:
                    yl, yu = op['angle_domain']
                else:
                    yl, yu = op['length_domain']

                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rand_var = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)

                if rand_var <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand_var + (1.0 - 2.0 * rand_var) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand_var) + 2.0 * (rand_var - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow

                y += deltaq * (yu - yl)
                y = np.clip(y, yl, yu)
                chromosome[i, j] = y

    return chromosome