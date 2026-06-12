import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from precalculate_all_orientation_segments import *
from decode_individual import *
from run_genetic_algorithm import *
from draw_problem_2d import *
from run_it_generic_hybrid import *
from initialize_random_population import *
from evaluate import *
from check_constraints import *
from ranking_evaluation import *
from selection import *
from variation import *
from survivor import *
from fix_angle import *
from calculate_single_orientation_segment import *



# 1. Set the generic font family ('serif', 'sans-serif', 'monospace')
plt.rcParams['font.family'] = 'serif'

# 2. Specify the exact font name
plt.rcParams['font.serif'] = ['Times New Roman']

try:
    import pandas as pd
except ImportError:
    print("---------------------------------------------------------------")
    print("Warning: 'pandas' library not found. Saving to .xlsx is disabled.")
    print("Please install it for Excel support: pip install pandas openpyxl")
    print("---------------------------------------------------------------")
    pd = None

# --- Core Import ---
try:
    from run_it_generic_hybrid import *
except ImportError:
    print("Error: Could not find 'run_it_generic_hybrid.py'.")
    print("Please make sure the main logic file is in the same directory.")
    exit()

# --- Task Setup Function ---
def experiment_runner(task_id):
    """
    Sets the 'op' (Optimization Problem) dictionary based on the task ID.
    """
    op = {}

    if task_id == 1:
        # task 1
        print("Setting up Task 1 (Wall)")
        op['targets'] = np.array([[110, 20, np.deg2rad(270)]])
        op['obstacles'] = np.array([
            [50, -40, 10], [50, -20, 10], [50, 0, 10], [50, 20, 10],
            [50, 40, 10], [50, 60, 10], [110, 110, 10]
        ])
        op['home_base'] = np.array([0, 20, 0])
        op['angle_domain'] = [-45, 45]
        op['length_domain'] = [5.0, 30.0]
        op['n_nodes'] = 8
        op['environment_bounds'] = [-20, 130, -60, 130]

    elif task_id == 2:
        # task 2
        print("Setting up Task 2 (Corridor)")
        op['targets'] = np.array([[150, 60, np.deg2rad(90)]])
        op['obstacles'] = np.array([
            [20, -40, 10], [20, -20, 10], [20, 0, 10], [20, 20, 10], [20, 40, 10], [20, 60, 10],
            [110, 110, 10], [110, 90, 10], [110, 70, 10], [110, 50, 10], [110, 30, 10], [150, -40, 10]
        ])
        op['home_base'] = np.array([-20, 0, np.deg2rad(90)])
        op['angle_domain'] = [-75, 75]
        op['length_domain'] = [5.0, 50.0]
        op['environment_bounds'] = [-50, 200, -50, 120]
        op['n_nodes'] = 10

    elif task_id == 4:
        # task 4
        print("Setting up Task 4 (Maze)")
        op['targets'] = np.array([[0, 170, np.deg2rad(90)]])
        op['obstacles'] = np.array([
            [-50, 0, 13], [0, 0, 13], [50, 0, 13],
            [-25, 50, 13], [25, 50, 13],
            [-50, 100, 13], [0, 100, 13], [50, 100, 13],
        ])
        op['home_base'] = np.array([0, -50, np.deg2rad(90)])
        op['environment_bounds'] = [-70, 70, -50, 180]
        op['n_nodes'] = 20
        op['angle_domain'] = [-65, 65]
        op['length_domain'] = [10.0, 70.0]

    elif task_id == 6:
        # task 6
        print("Setting up Task 6 (Scattered)")
        op['targets'] = np.array([[-50, 60, np.deg2rad(170)]])
        op['obstacles'] = np.array([
            [-35, 80, 14], [-30, 20, 14], [10, 50, 14], [35, 30, 14],
        ])
        op['home_base'] = np.array([50, 0, np.deg2rad(90)])
        op['environment_bounds'] = [-70, 70, 0, 120]
        op['n_nodes'] = 20
        op['angle_domain'] = [-75, 75]
        op['length_domain'] = [10.0, 70.0]

    elif task_id == 7:
        # Task 7
        print("Setting up Task 7 (Simple)")
        op['targets'] = np.array([[100, 30, np.deg2rad(-10)]])
        op['obstacles'] = np.array([
            [50, -20, 10], [40, 30, 10], [60, 10, 10], [100, -10, 10]
        ])
        op['home_base'] = np.array([0, 0, 0])
        op['n_nodes'] = 20
        op['angle_domain'] = [-45, 45]
        op['length_domain'] = [10, 40]
        op['environment_bounds'] = [-10, 150, -40, 80]

    elif task_id == 8:
        # Task 8
        print("Setting up Task 8 (Soft Robot Task)")
        op['targets'] = np.array([ 
                [100, -45, np.deg2rad(-45) ]
                ])
        op['obstacles'] = np.array([
                [50, -20, 10],
                [40, 30, 10],
                [60, 10, 10],
                [100, -10, 10]
                ])
        op['home_base'] = np.array([0, 0, 0])
        op['n_nodes'] = 20
        op['angle_domain'] = [-45, 45]
        op['length_domain'] = [10, 40]
        op['environment_bounds'] = [-20, 140, -60, 80]
    
    elif task_id == 9:
        # Task 9
        print("Setting up Task 9")
        op['targets'] = np.array([ 
            [170, 110, np.deg2rad(45)]
            ])
        op['obstacles'] = np.array([

            # x   y   r
            # [0, 10, 10],
            [0, 50, 10],
            [0, 80, 10],


            [40, 10, 10],
            [40, 40, 10],
            [40, 70, 10],
            [40, 100, 10],

            # [70, 10, 10],

            [90, 32, 10],
            [90, 70, 10],
            # [90, 87, 10],
            [90, 110, 10],




            # [90, 30, 10],
            # [110, 30, 10],
            [130, 30, 10],


            [120, 110, 10],
            [130, 80, 10],
            [143, 55, 10],
            [155, 10, 10],
            
            ])
        op['home_base'] = np.array([0, 0, np.deg2rad(90)])
        op['n_nodes'] = 20
        op['environment_bounds'] = ([-50, 200, 0, 130])
        op['angle_domain'] = [-45, 45]
        op['length_domain'] = [10.0, 40.0]
    
    else:
        raise ValueError(f"Task ID {task_id} not defined.")
    
    op['robot_radius'] = 1.0
    op['end_points'] = precalculate_all_orientation_segments(op)
    print(f"--- Task {task_id} Set Up ---")

    return op


# --- GA Settings Function ---

def setup_ga_settings(max_evals, pop_size, permutation_id):
    """
    Sets the 'gas' (Genetic Algorithm Settings) dictionary.
    Decoupled: Takes explicit max_evals and pop_size arguments.
    """
    gas = {}

    # --- Decoupled Settings ---
    gas['max_evaluations'] = max_evals
    gas['n_individuals'] = pop_size
    gas['ranking_permutation'] = permutation_id

    # --- Base Settings ---
    gas['penalty_method'] = 'static'
    gas['selection_method'] = 'tournament'
    gas['obstacle_avoidance'] = True
    
    gas['crossover_method'] = 'blxa'
    gas['crossover_probability'] = 1.0
    gas['crossover'] = {'alpha': 0.5, 'eta': 20}

    gas['mutation_method'] = 'random'
    gas['mutation_probability'] = 0.6
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
        'ikFitnessModified': 0, 'pathLengthModified': 1, 'ikFitness': 2,
        'nodeCount': 3, 'undulation': 4, 'pathLength': 5,
        'ornError': 6, 'penalty': 7, 'rank': 8, 'id': 9,
    }
    gas['variance_generations'] = 20
    gas['stopAtVariance_flag'] = False
    gas['stopAtVariance_zeros'] = 5

    # --- DYNAMIC CALCULATION ---
    # Calculates generations so that Generations * PopSize ~= MaxEvaluations
    gas['generations'] = int(np.ceil(gas['max_evaluations'] / gas['n_individuals']))
    
    print(f"--- GA Settings Set Up ---")
    print(f"Max Evaluations: {gas['max_evaluations']}")
    print(f"Population: {gas['n_individuals']}")
    print(f"Calculated Max Generations: {gas['generations']}")
    print(f"Ranking Permutation: {gas['ranking_permutation']}")

    return gas


# --- Main Experiment Runner ---

def run_experiment(task_id, max_evals, pop_size, num_runs, permutation_id, save_results=True):
    """
    Runs a full experiment for a specific Task, FE limit, and Population Size.
    """
    
    # Setup problem and GA settings
    op = experiment_runner(task_id)
    gas = setup_ga_settings(max_evals, pop_size, permutation_id)
    
    print(f"\n===== STARTING EXPERIMENT =====\n"
          f"Task ID: {task_id}\n"
          f"Max Evals: {gas['max_evaluations']}\n"
          f"Population: {gas['n_individuals']}\n"
          f"Permutation: {permutation_id}\n"
          f"Num Runs: {num_runs}\n"
          f"=============================")

    # --- Data storage for all runs ---
    all_best_chromosomes = []
    num_metrics = len(gas['fitIdx'])
    all_best_fitness_arrays = np.zeros((num_runs, num_metrics))
    all_run_times = np.zeros(num_runs)
    
    # Create a directory for results using the requested format logic
    # Directory: taskX_FEX_PopX/Permutation_Y
    base_dir = f"task{task_id}_FE{max_evals}_Pop{pop_size}"
    results_dir = os.path.join(base_dir, f"Permutation_{permutation_id}")
    
    if save_results and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    total_start_time = time.time()

    for r in range(num_runs):
        print(f'\n--- Starting Run {r + 1} of {num_runs} ---')
        run_start_time = time.time()
        
        # Run the core GA
        population, ranked_fit_array, _ = run_genetic_algorithm(op, gas, r + 1)
        
        run_time = time.time() - run_start_time

        # Store best of this run
        best_id_this_run = int(ranked_fit_array[0, gas['fitIdx']['id']])
        best_chrom_this_run = population[:, :, best_id_this_run]
        best_fitness_this_run = ranked_fit_array[0, :]
        
        all_best_chromosomes.append(best_chrom_this_run)
        all_best_fitness_arrays[r, :] = best_fitness_this_run
        all_run_times[r] = run_time
        
        print(f"Run {r+1} finished in {run_time:.2f}s. "
              f"Best IK: {best_fitness_this_run[gas['fitIdx']['ikFitness']]:.4f}")

        # Save plot for this run's best
        if save_results:
            paths_xy, _ = decode_individual(op, best_chrom_this_run)
            
            # --- SAVE PATH COORDINATES & CHROMOSOME ---
            path_coords = paths_xy[0] # XY coordinates (N x 2)
            
            coord_file = f"{results_dir}/Run_{r+1}_coords.csv"
            chrom_file = f"{results_dir}/Run_{r+1}_chromosome.npy"
            
            # Save Coordinates (CSV)
            if pd is not None:
                pd.DataFrame(path_coords, columns=['x', 'y']).to_csv(coord_file, index=False)
            else:
                np.savetxt(coord_file, path_coords, delimiter=",", header="x,y", comments='')
            
            # Save Chromosome (NPY)
            np.save(chrom_file, best_chrom_this_run)
            print(f"  -> Saved coordinates to {coord_file}")
            # ------------------------------------------

            plt.figure(figsize=(19.2, 10.8))
            draw_problem_2d(op, gas, paths_xy, best_fitness_this_run) 
            # plt.title(f"Run {r+1} Best | Task {task_id}, FE {max_evals}, Pop {pop_size}, Perm {permutation_id}", color='black', fontsize=33, wrap=True)
            fig_path = f"{results_dir}/Run_{r+1}.pdf"
            plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    total_time = time.time() - total_start_time
    print('\n--- All Runs Completed ---')
    print(f'Total experiment time: {total_time:.2f} seconds')

    # --- Find the best overall run ---
    sorted_indices = np.lexsort((
        all_best_fitness_arrays[:, gas['fitIdx']['ikFitness']],
        all_best_fitness_arrays[:, gas['fitIdx']['rank']]
    ))
    best_run_idx = sorted_indices[0]

    best_overall_chromosome = all_best_chromosomes[best_run_idx]
    best_overall_fitness = all_best_fitness_arrays[best_run_idx, :]

    print(f'\nBest overall result found in run {best_run_idx + 1}.')

    # Plot the best overall solution
    paths_xy, _ = decode_individual(op, best_overall_chromosome)
    plt.figure(figsize=(19.2, 10.8))
    draw_problem_2d(op, gas, paths_xy, best_overall_fitness)

    # # title_str = (f"BEST OVERALL (Run {best_run_idx + 1}) | Task {task_id}, FE {max_evals}, Pop {pop_size}, Perm {permutation_id}\n"
    # title_str =   (f"F1 (TD): {best_overall_fitness[2]:.2f}, "
    #             f"F2 (Len): {best_overall_fitness[5]:.1f}, "
    #             f"F3 (Und): {best_overall_fitness[4]:.2f}, "
    #             f"F4 (# Seg): {int(round(best_overall_fitness[3]))}")
    # plt.title(title_str, color='black', fontsize=33, wrap=True)
    # plt.tight_layout()
    
    if save_results:
        fig_filename = f"{results_dir}/_BEST_OVERALL.pdf"
        plt.savefig(fig_filename, bbox_inches='tight', pad_inches=0.5)
        print(f"Saved best overall plot to {fig_filename}")

    # Save all numerical results to a file
    if save_results:
        # --- COMBINE RESULTS ---
        run_times_col = all_run_times.reshape(-1, 1)
        results_to_save = np.hstack((all_best_fitness_arrays, run_times_col))
        
        results_file = f"{results_dir}/_results_summary.npy"
        np.save(results_file, results_to_save)
        
        # --- Save to XLSX with requested naming format ---
        header_list = list(gas['fitIdx'].keys()) + ["runTime"]

        if pd is not None:
            try:
                # Format: results_summary_taskX_FEX_PopX_ga_pX.xlsx
                # Interpreting 'pX' as Population Size as per standard convention or redundant labeling
                xlsx_filename = f"{results_dir}/results_summary_task{task_id}_FE{max_evals}_Pop{pop_size}_perm{permutation_id}.xlsx"
                
                print(f"Attempting to save results to {xlsx_filename}...")
                df = pd.DataFrame(results_to_save, columns=header_list)
                df.to_excel(xlsx_filename, index=False, engine='openpyxl')
                print(f"Successfully saved results summary to {xlsx_filename}")
            
            except Exception as e:
                print(f"--- ERROR: Could not save to .xlsx file: {e}")
        else:
            print("--- INFO: 'pandas' not found, skipping Excel save. ---")


# --- Main execution block ---
if __name__ == '__main__':
    
    # --- EXPERIMENT CONFIGURATION ---

    TASKS_TO_RUN = [4]          # Tasks to Execute
    
    # 3x3 Matrix Configuration
    # FE_VALUES = [250]   # Function Evaluation Levels
    # POP_VALUES = [100]         # Population Size Levels
    # PERMUTATIONS = [1]    # Ranking Permutations




    FE_VALUES = [250]   # Function Evaluation Levels
    POP_VALUES = [500]         # Population Size Levels
    PERMUTATIONS = [6]    # Ranking Permutations
    


    NUMBER_OF_RUNS = 1                   # Runs per combination
    # NUMBER_OF_RUNS = 2
                       # Runs per combination
    
    print(f"===== STARTING 3x3 MATRIX EXPERIMENT BATCH =====")
    print(f"Tasks: {TASKS_TO_RUN}")
    print(f"Function Evaluations: {FE_VALUES}")
    print(f"Population Sizes: {POP_VALUES}")
    print(f"Permutations: {PERMUTATIONS}")
    print(f"Runs per combination: {NUMBER_OF_RUNS}")
    print("================================================")
    
    # Loop through Task -> FE -> Pop (The 3x3 Matrix Logic) -> Permutations
    for task_id in TASKS_TO_RUN:
        for fe in FE_VALUES:
            for pop in POP_VALUES:
                for perm_id in PERMUTATIONS:
                    
                    print(f"\n\n-------------------------------------------------")
                    print(f"--- Starting: TASK {task_id} | FE {fe} | POP {pop} | PERM {perm_id} ---")
                    print(f"-------------------------------------------------\n")
                    
                    run_experiment(
                        task_id=task_id,
                        max_evals=fe,
                        pop_size=pop,
                        num_runs=NUMBER_OF_RUNS,
                        permutation_id=perm_id,
                        save_results=True
                    )
                    
                    print(f"\n--- Completed: TASK {task_id} | FE {fe} | POP {pop} | PERM {perm_id} ---")

    print("\n\n=======================================")
    print("===== ALL EXPERIMENT BATCHES COMPLETE =====")
    print("=======================================\n")