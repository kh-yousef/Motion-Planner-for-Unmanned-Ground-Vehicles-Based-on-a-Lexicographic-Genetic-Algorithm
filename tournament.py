import numpy as np
def tournament(gas, fit_array_cols, k):
    """
    Tournament selection implementation.
    Arguments:
        fit_array_cols: Nx2 or more array, columns include rank and id
        k: tournament size (number of competitors per selection)
    Returns:
        mat_pool: Array of selected individual IDs
    """
    n_individuals = fit_array_cols.shape[0]
    mat_pool = np.zeros(gas['n_individuals'], dtype=int)

    for i in range(gas['n_individuals']):
        best_rank = np.inf
        winner_id = -1
        for _ in range(k):
            competitor_idx = np.random.randint(0, n_individuals)
            competitor_rank = fit_array_cols[competitor_idx, 0]
            if competitor_rank < best_rank:
                best_rank = competitor_rank
                winner_id = int(fit_array_cols[competitor_idx, 1])
        mat_pool[i] = winner_id
    return mat_pool
