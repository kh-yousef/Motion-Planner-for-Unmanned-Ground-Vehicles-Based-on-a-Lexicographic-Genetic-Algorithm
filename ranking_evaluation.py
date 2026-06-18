import numpy as np

def ranking_evaluation(gas, fit_array):

    # 1. Modulo Arithmetic Partitioning
    step_ik = gas['ranking']['step_ik']
    step_path_len = gas['ranking'].get('step_path_len', 5) # Fallback in case step_path_len isn't explicitly defined

    
    fit_array[:, gas['fitIdx']['ikFitnessModified']] = fit_array[:, gas['fitIdx']['ikFitness']] - (fit_array[:, gas['fitIdx']['ikFitness']] % step_ik)
    fit_array[:, gas['fitIdx']['pathLengthModified']] = fit_array[:, gas['fitIdx']['pathLength']] - (fit_array[:, gas['fitIdx']['pathLength']] % step_path_len)

    # Extract columns for readability 
    f_ik_mod = fit_array[:, gas['fitIdx']['ikFitnessModified']]
    f_path_mod = fit_array[:, gas['fitIdx']['pathLengthModified']]
    f_node = fit_array[:, gas['fitIdx']['nodeCount']]
    f_und = fit_array[:, gas['fitIdx']['undulation']]

    # 2. Initial Base Sort
    
    sort_tuple_initial = (f_path_mod, f_und, f_node, f_ik_mod)
    initial_order = np.lexsort(sort_tuple_initial)
    fit_array = fit_array[initial_order]

    # 3. Boundary Detection (diff_array)
    diff_array = np.zeros(fit_array.shape[0])
    cols_to_check = [
        gas['fitIdx']['ikFitnessModified'], 
        gas['fitIdx']['nodeCount'], 
        gas['fitIdx']['undulation'], 
        gas['fitIdx']['pathLengthModified']
    ]
    
    # Calculate absolute differences between consecutive rows
    for i in range(1, fit_array.shape[0]):
        diff_array[i] = np.sum(np.abs(fit_array[i-1, cols_to_check] - fit_array[i, cols_to_check]))

    # 4. Two-Pass Block Sub-Sorting 
    start = 0
    for i in range(fit_array.shape[0]):
        if diff_array[i] > 0:
            stop = i
            block = fit_array[start:stop]
            
            # Sub-sort the isolated block using raw continuous values
            b_ik_raw = block[:, gas['fitIdx']['ikFitness']]
            b_path_raw = block[:, gas['fitIdx']['pathLength']]
            
            # priority: [ikFitness, pathLength]. 
            block_order = np.lexsort((b_path_raw, b_ik_raw))
            fit_array[start:stop] = block[block_order]
            
            start = stop

    # 5. Final Rank Assignment
    fit_array[:, gas['fitIdx']['rank']] = np.arange(1, fit_array.shape[0] + 1)

    # 6. Update ranking stats
    first_ikmod_val = fit_array[0, gas['fitIdx']['ikFitnessModified']]
    gas['ranking']['firstPartitionSize'] = np.sum(fit_array[:, gas['fitIdx']['ikFitnessModified']] == first_ikmod_val)
    gas['ranking']['minFit'] = first_ikmod_val

    return fit_array