from tournament import *
def selection(gas, fit_array):
    """
    Perform selection of parents for reproduction based on configured method.
    Currently supports tournament selection of size 2.
    Arguments:
        fit_array: Array of fitness data with rank and id columns
    Returns:
        mat_pool: Array of selected individual IDs for mating pool
    """
    if gas['selection_method'] == 'tournament':
        return tournament(gas, fit_array, 2)
    else:
        raise RuntimeError("Unexpected selection method.")