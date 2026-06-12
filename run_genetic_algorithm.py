import numpy as np
from initialize_random_population import *
from evaluate import *
from check_constraints import *
from ranking_evaluation import *
from selection import *
from variation import *
from survivor import *
from precalculate_all_orientation_segments import *
from decode_individual import *
from run_it_generic_hybrid import *
from draw_problem_2d import *
from fix_angle import *
from calculate_single_orientation_segment import *
def run_genetic_algorithm(op, gas, exp):
    """
    Core genetic algorithm loop.
    Implements a FIXED FUNCTION EVALUATION budget.
    The number of generations scales dynamically: Generations = MaxEvaluations / PopSize.
    """
    queue = np.zeros(gas['variance_generations'])
    q_index = 0
    best_fitness_history = []
    
    # Initialize counters
    evaluations_count = 0
    generation_count = 0

    # Ensure even-sized populations for pairing in variation step
    if gas['n_individuals'] % 2 != 0:
        gas['n_individuals'] += 1

    print(f'\n--- Initializing Population (Target Evals: {gas["max_evaluations"]}, Pop: {gas["n_individuals"]}) ---')
    population = initialize_random_population(op, gas)
    population, fit_array_P = evaluate(op, gas, population)
    
    # Update count after initialization
    evaluations_count += gas['n_individuals']
    
    fit_array_P = check_constraints(op, gas, population, fit_array_P)
    fit_array_P = ranking_evaluation(gas, fit_array_P)

    best_fitness_history.append(fit_array_P[0, gas['fitIdx']['ikFitness']])
    print(f"Best IK Fitness in Gen 0: {best_fitness_history[0]:.4f}")
    print('--- Starting Evolution ---')

    # --- LOOP CONDITION: Based strictly on Function Evaluations ---
    while evaluations_count < gas['max_evaluations']:
        generation_count += 1
        
        # 1. Selection
        mat_pool = selection(gas, fit_array_P[:, [gas['fitIdx']['rank'], gas['fitIdx']['id']]])
        
        # 2. Variation (Crossover & Mutation)
        offspring = variation(op, gas, population, mat_pool)
        
        # 3. Evaluation
        offspring, fit_array_O = evaluate(op, gas, offspring)
        
        # Update evaluations count (Cost = Population Size)
        evaluations_count += gas['n_individuals']
        
        # 4. Constraints & Survival
        fit_array_O = check_constraints(op, gas, offspring, fit_array_O)
        population, fit_array_P = survivor(gas, population, offspring, fit_array_P, fit_array_O)

        # Logging & Convergence Check
        best_fitness_history.append(fit_array_P[0, gas['fitIdx']['ikFitness']])
        queue[q_index % gas['variance_generations']] = fit_array_P[0, gas['fitIdx']['ikFitness']]
        q_index += 1

        variance = np.var(queue) if generation_count >= gas['variance_generations'] else float('inf')
        if gas['verbose']:
            best_fit = fit_array_P[0, :]
            print(f"[{exp}.G{generation_count} | Eval {evaluations_count}/{gas['max_evaluations']}] "
                  f"Rank: {int(best_fit[gas['fitIdx']['rank']])}\t"
                  f"(IK: {best_fit[gas['fitIdx']['ikFitness']]:.2f}, "
                  f"Len: {best_fit[gas['fitIdx']['pathLength']]:.1f}, "
                  f"Pen: {best_fit[gas['fitIdx']['penalty']]:.0f})\t"
                  f"Var: {variance:.6f}")

        # Stopping criteria: Variance (Optional)
        if (gas['stopAtVariance_flag'] and generation_count > gas['variance_generations']):
            if round(variance, gas['stopAtVariance_zeros']) == 0:
                print(f'Evolution stopped: Solution has converged (variance 0) at Eval {evaluations_count}.')
                break

    if evaluations_count >= gas['max_evaluations']:
        print(f'Evolution finished: Reached max evaluations ({evaluations_count}).')

    return population, fit_array_P, best_fitness_history