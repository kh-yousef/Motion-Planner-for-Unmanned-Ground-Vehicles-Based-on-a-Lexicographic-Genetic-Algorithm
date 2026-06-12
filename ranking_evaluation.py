import numpy as np
def ranking_evaluation(gas, fit_array):
    """
    Rank the population fitness array to assign ranks based on multiple fitness criteria.
    Dynamically sorts based on gas['ranking_permutation'].
    
    np.lexsort((secondary, primary)) -> The last key in the tuple is the PRIMARY sort key.
    """
    # Separate finite and infinite fitness individuals based on IK fitness
    finite_mask = np.isfinite(fit_array[:, gas['fitIdx']['ikFitness']])
    finite_fit = fit_array[finite_mask]
    infinite_fit = fit_array[~finite_mask]

    if finite_fit.size > 0:
        # Discretize IK fitness for ranking bins
        finite_fit[:, gas['fitIdx']['ikFitnessModified']] = np.floor(
            finite_fit[:, gas['fitIdx']['ikFitness']] / gas['ranking']['step_ik']) * gas['ranking']['step_ik']
        
        # Get permutation setting, default to 1
        perm_id = gas.get('ranking_permutation', 1)
        
        # Define fields
        # Note: tie breakers (pathLength, ikFitness) are usually least significant in lexsort tuple
        # if we want them to break ties when everything else is equal.
        # So they go FIRST in the lexsort tuple.
        
        f_path_raw = finite_fit[:, gas['fitIdx']['pathLength']]
        f_ik_raw   = finite_fit[:, gas['fitIdx']['ikFitness']]
        
        f_path_mod = finite_fit[:, gas['fitIdx']['pathLengthModified']]
        f_und      = finite_fit[:, gas['fitIdx']['undulation']]
        f_node     = finite_fit[:, gas['fitIdx']['nodeCount']]
        f_ik_mod   = finite_fit[:, gas['fitIdx']['ikFitnessModified']]

        # Construct Tuple: (Least Significant, ..., Most Significant)
        # We always keep IK Modified as Most Significant (Last in tuple)
        
        if perm_id == 1:
            # Permutation 1: IK > Path > Und > Node
            # Tuple: (Node, Und, Path, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_node, f_und, f_path_mod, f_ik_mod)
            
        elif perm_id == 2:
            # Permutation 2: IK > Path > Node > Und
            # Tuple: (Und, Node, Path, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_und, f_node, f_path_mod, f_ik_mod)
            
        elif perm_id == 3:
            # Permutation 3: IK > Node > Path > Und
            # Tuple: (Und, Path, Node, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_und, f_path_mod, f_node, f_ik_mod)
            
        elif perm_id == 4:
            # Permutation 4: IK > Node > Und > Path
            # Tuple: (Path, Und, Node, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_path_mod, f_und, f_node, f_ik_mod)
            
        elif perm_id == 5:
            # Permutation 5: IK > Und > Path > Node
            # Tuple: (Node, Path, Und, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_node, f_path_mod, f_und, f_ik_mod)
            
        elif perm_id == 6:
            # Permutation 6: IK > Und > Node > Path
            # Tuple: (Path, Node, Und, IK)
            sort_tuple = (f_path_raw, f_ik_raw, f_path_mod, f_node, f_und, f_ik_mod)
            
        else:
            print(f"Warning: Unknown Permutation ID {perm_id}, defaulting to 1")
            sort_tuple = (f_path_raw, f_ik_raw, f_node, f_und, f_path_mod, f_ik_mod)

        # Sort
        sort_order = np.lexsort(sort_tuple)
        finite_fit = finite_fit[sort_order]

    # Recombine finite and infinite fitness arrays
    ranked_fit = np.vstack((finite_fit, infinite_fit)) if infinite_fit.size > 0 else finite_fit

    # Assign ranks: starting from 1 to population size
    ranked_fit[:, gas['fitIdx']['rank']] = np.arange(1, ranked_fit.shape[0] + 1)

    return ranked_fit