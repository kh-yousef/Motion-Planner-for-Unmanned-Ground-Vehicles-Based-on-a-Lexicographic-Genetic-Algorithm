import numpy as np

def perform_gaussian_mutation(op, gas, chromosome):
    """
    Gaussian mutation applied gene-wise based on mutation probability and gene-swap probability.
    """
    n_targets = chromosome.shape[0] - 1

    for i in range(n_targets + 1):
        for j in range(op['n_nodes']):
            if np.random.rand() < gas['mutation']['gene_swap_probability']:
                current_value = chromosome[i, j]
                if i < n_targets:
                    bounds = op['angle_domain']
                    sigma = gas['mutation']['sigma_angle']
                else:
                    bounds = op['length_domain']
                    sigma = gas['mutation']['sigma_length']

                mutation_value = np.random.randn() * sigma
                new_value = current_value + mutation_value
                new_value = np.clip(new_value, bounds[0], bounds[1])
                chromosome[i, j] = new_value

    return chromosome
