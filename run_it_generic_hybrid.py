import numpy as np
import matplotlib.pyplot as plt
import time
from initialize_random_population import *
from evaluate import *
from check_constraints import *
from ranking_evaluation import *
from selection import *
from variation import *
from survivor import *
from precalculate_all_orientation_segments import *
from decode_individual import *
from run_genetic_algorithm import *
from draw_problem_2d import *
from fix_angle import *
from calculate_single_orientation_segment import *




def run_it_generic_hybrid():
    """
    Main entry function to run the Genetic Algorithm for wheeled mobile robot motion planning.
    Sets problem parameters, GA parameters, runs multiple GA runs, and produces best solution.
    All task definitions and comments from the original MATLAB are preserved.
    """
    print('--- Mobile Robot Motion Planning with a Genetic Algorithm (Hybrid Strategy) ---')
    start_time = time.time()

    # --- Create local op and gas dictionaries ---
    op = {}
    gas = {}

    print('Setting up problem definition...')

    # --- Tasks Definitions (Uncomment to activate a task) ---
    # (All task definitions preserved as per your code block — omitted here for brevity)


    #  # # task 1
    # op['targets'] = np.array([[110, 20, np.deg2rad(270)]])
    # op['obstacles'] = np.array([
    #     [50, -40, 10],
    #     [50, -20, 10],
    #     [50, 0, 10],
    #     [50, 20, 10],
    #     [50, 40, 10],
    #     [50, 60, 10],
    #     [110, 110, 10]
    # ])
    # op['home_base'] = np.array([0, 20, 0])
    # op['angle_domain'] = [-45, 45]
    # op['length_domain'] = [5.0, 30.0]
    # op['n_nodes'] = 8

    # # task 2
    # op['targets'] = np.array([[150, 60, np.deg2rad(90)]])
    # op['obstacles'] = np.array([
    #     [20, -40, 10], [20, -20, 10], [20, 0, 10], [20, 20, 10], [20, 40, 10], [20, 60, 10],
    #     [110, 110, 10], [110, 90, 10], [110, 70, 10], [110, 50, 10], [110, 30, 10], [150, -40, 10]
    # ])
    # op['home_base'] = np.array([-20, 0, np.deg2rad(90)])
    # op['angle_domain'] = [-75, 75]
    # op['length_domain'] = [5.0, 50.0]
    # op['environment_bounds'] = [-50, 200, -50, 120]  # [x_min, x_max, y_min, y_max]
    # op['n_nodes'] = 10

#######################################################################################
################################used for experiment####################################

    # task 4
    # op['targets'] = np.array([[0, 170, np.deg2rad(90)]])
    # op['obstacles'] = np.array([
    #     [-50, 0, 13], [0, 0, 13], [50, 0, 13],
    #     [-25, 50, 13], [25, 50, 13],
    #     [-50, 100, 13], [0, 100, 13], [50, 100, 13],
        
    # ])
    # op['home_base'] = np.array([0, -50, np.deg2rad(90)])
    # op['environment_bounds'] = [-70, 70, -50, 180]
    # op['n_nodes'] = 20
    # op['angle_domain'] = [-65, 65]
    # op['length_domain'] = [10.0, 70.0]



    # # task 6
    # op['targets'] = np.array([[-50, 60, np.deg2rad(170)]])
    # op['obstacles'] = np.array([
    #     [-35, 80, 14], [-30, 20, 14], [10, 50, 14], [35, 30, 14],
        
    # ])
    # op['home_base'] = np.array([50, 0, np.deg2rad(90)])
    # op['environment_bounds'] = [-70, 70, 0, 120]
    # op['n_nodes'] = 20
    # op['angle_domain'] = [-75, 75]
    # op['length_domain'] = [10.0, 70.0]

    # Task 7 (Active)
    # op['targets'] = np.array([
    #     [100, 30, np.deg2rad(-10)]
    # ])
    # op['obstacles'] = np.array([
    #     [50, -20, 10],
    #     [40, 30, 10],
    #     [60, 10, 10],
    #     [100, -10, 10]
    # ])
    # op['home_base'] = np.array([0, 0, 0])
    # op['n_nodes'] = 20
    # op['angle_domain'] = [-45, 45]
    # op['length_domain'] = [10, 40]
    # op['environment_bounds'] = [-10, 150, -40, 80]


    # #soft robot tasks
    op['home_base'] = np.array([0, 0, 0])
    op['targets'] = np.array([ 
                  [100, -45, np.deg2rad(-45) ]
                #   [120, -40, np.deg2rad(-30) ]
                #   [100, 30, np.deg2rad(-10)]
                #   [100, 50, np.deg2rad(45)]
                  ])
    op['obstacles'] = np.array([
                    [50, -20, 10],
                    [40, 30, 10],
                    [60, 10, 10],
                    [100, -10, 10]
                    ])
    op['n_nodes'] = 20
    op['angle_domain'] = [-45, 45]
    op['length_domain'] = [10, 40]
    op['environment_bounds'] = [-20, 140, -60, 80]

    # Robot Radius
    op['robot_radius'] = 1.0

    # Pre-calculate target orientation segments
    # Pass 'op' as an argument
    op['end_points'] = precalculate_all_orientation_segments(op)

    print('Setting up GA parameters...')
    gas['num_runs'] = 1
    gas['max_evaluations'] = 50000
    gas['n_individuals'] = 100
    gas['penalty_method'] = 'static'
    gas['selection_method'] = 'tournament'
    gas['obstacle_avoidance'] = True

    gas['crossover_method'] = 'blxa'
    gas['crossover_probability'] = 1.0
    gas['crossover'] = {'alpha': 0.5, 'eta': 20}

    gas['mutation_method'] = 'random'
    gas['mutation_probability'] = 0.3
    gas['mutation'] = {
        'eta': 20,
        'gene_swap_probability': 0.2,
        'sigma_angle': 10.0,
        'sigma_length': 5.0
    }
    
    gas['survival_method'] = 'elitist'
    gas['verbose'] = True

    gas['enforce_orientation'] = True
    gas['extra_genes'] = 4

    gas['constraints'] = {
        'collision_weight': 100,
        'orientation_weight': 10,
        'angle_bound_weight': 10
    }
    gas['fitness_strategy'] = 'weighted_distance'
    gas['ranking'] = {'step_ik': 0.5, 'step_path_len': 5.0}

    gas['fitIdx'] = {
        'ikFitnessModified': 0,
        'pathLengthModified': 1,
        'ikFitness': 2,
        'nodeCount': 3,
        'undulation': 4,
        'pathLength': 5,
        'ornError': 6,
        'penalty': 7,
        'rank': 8,
        'id': 9,
    }

    gas['variance_generations'] = 20
    gas['stopAtVariance_flag'] = True
    gas['stopAtVariance_zeros'] = 5

    all_best_chromosomes = []
    all_best_fitness_arrays = np.zeros((gas['num_runs'], len(gas['fitIdx'])))

    for r in range(gas['num_runs']):
        print(f'\n--- Starting Run {r + 1} of {gas["num_runs"]} ---')
        # Pass op and gas
        population, ranked_fit_array, _ = run_genetic_algorithm(op, gas, r + 1)

        best_id_this_run = int(ranked_fit_array[0, gas['fitIdx']['id']])
        all_best_chromosomes.append(population[:, :, best_id_this_run])
        all_best_fitness_arrays[r, :] = ranked_fit_array[0, :]

    sorted_indices = np.lexsort((
        all_best_fitness_arrays[:, gas['fitIdx']['ikFitness']],
        all_best_fitness_arrays[:, gas['fitIdx']['rank']]
    ))
    best_run_idx = sorted_indices[0]

    best_overall_chromosome = all_best_chromosomes[best_run_idx]
    best_overall_fitness = all_best_fitness_arrays[best_run_idx, :]

    total_time = time.time() - start_time
    print('\n--- All Runs Completed ---')
    print(f'Best result found in run {best_run_idx + 1}.')
    print(f'Total execution time: {total_time:.2f} seconds\n')

    # Pass op
    paths_xy, _ = decode_individual(op, best_overall_chromosome)


    
    # Add this line to debug
    print("Decoded paths:", paths_xy) 
    
  


    print('Generating final plot for the best overall solution...')
    # Pass op and gas
    draw_problem_2d(op, gas, paths_xy, best_overall_fitness)

    print('--- Script Finished ---')
    return best_overall_chromosome, paths_xy


if __name__ == '__main__':
    run_it_generic_hybrid()