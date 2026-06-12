from elitist_survival import *
def survivor(gas, pop, offspring, fit_array_P, fit_array_O):
    """
    Perform survivor selection according to the configured method.
    Elitist method combines parents and offspring, ranks and selects top individuals.
    """
    if gas['survival_method'] == 'elitist':
        return elitist_survival(gas, pop, offspring, fit_array_P, fit_array_O)
    else:
        raise RuntimeError('Unexpected survival method.')