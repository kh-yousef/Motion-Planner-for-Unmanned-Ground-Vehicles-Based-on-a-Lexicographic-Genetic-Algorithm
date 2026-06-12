from decode_individual import *
from calculate_segment_circle_intersection_depth import *
from fix_angle import *

def calculate_static_penalty(op, gas, chromosome):
    """
    Calculate the static penalty for a chromosome based on collisions, angle bounds, and orientation.
    Proportional penalties are applied and environment boundary violations cause infinite penalty.
    """
    if 'obstacle_avoidance' in gas and not gas['obstacle_avoidance']:
        return 0

    total_penalty = 0


    paths_xy, _ = decode_individual(op, chromosome)

    # 1. Proportional collision penalty
    if 'obstacle_avoidance' in gas and gas['obstacle_avoidance']:
        total_intersection_depth = 0
        if 'obstacles' in op and len(op['obstacles']) > 0:
            for path in paths_xy:
                for j in range(len(path) - 1):
                    p1, p2 = path[j], path[j+1]
                    for obstacle in op['obstacles']:
                        # NOTE: Assumes a calculate_segment_circle_intersection_depth function exists
                        depth = calculate_segment_circle_intersection_depth(op, p1, p2, obstacle)
                        total_intersection_depth += depth
        total_penalty += total_intersection_depth * gas['constraints']['collision_weight']

    total_penalty += total_intersection_depth * gas['constraints']['collision_weight']

    # 2. Proportional final angle bounds penalty
# --- START MODIFICATION 2: Added Orientation Penalty ---
    total_angle_bound_violation = 0
    total_orientation_error_violation = 0
    orientation_tolerance_deg = 10.0  # Your requested +/- 10 degree tolerance
    
    ntargets = chromosome.shape[0] - 1
    if ntargets < 1: ntargets = 1
        
    # Define gene indices
    n_nodes = op['n_nodes']
    IDX_FINAL_SEG_IDX = n_nodes
    IDX_FINAL_ANGLE = n_nodes + 1
    
    for i in range(ntargets):
        # --- A. Final Angle Bounds Penalty (Existing) ---
        final_alignment_angle = chromosome[i, IDX_FINAL_ANGLE]
        min_angle, max_angle = op['angle_domain']
        angle_bound_violation = abs(min(0, final_alignment_angle - min_angle)) + abs(min(0, max_angle - final_alignment_angle))
        total_angle_bound_violation += angle_bound_violation

        # --- B. Final Orientation Penalty (New) ---
        
        # 1. Get Target's final orientation
        target_orientation_deg = np.degrees(op['targets'][i][2])

        # 2. Calculate the Robot's final orientation
        initial_orientation_deg = np.degrees(op['home_base'][2])
        final_segment_idx = int(round(chromosome[i, IDX_FINAL_SEG_IDX]))
        
        # Sum all angles *before* the final cut
        sum_of_path_angles_deg = np.sum(chromosome[i, :final_segment_idx])
        
        # Get the final "docking" angle
        final_turn_angle_deg = chromosome[i, IDX_FINAL_ANGLE]
        
        # Robot's final orientation is the sum of all turns
        robot_final_orientation_deg = initial_orientation_deg + sum_of_path_angles_deg + final_turn_angle_deg

        # 3. Calculate the normalized error
        orientation_error_deg = robot_final_orientation_deg - target_orientation_deg
        normalized_error_deg = fix_angle(orientation_error_deg) # Crucial step!
        
        # 4. Calculate violation outside the +/- 10 degree tolerance
        orientation_violation = max(0, abs(normalized_error_deg) - orientation_tolerance_deg)
        total_orientation_error_violation += orientation_violation

    # Add both penalties to the total
    total_penalty += total_angle_bound_violation * gas['constraints']['angle_bound_weight']
        # --- MODIFICATION: Only apply orientation penalty if enforce_orientation is TRUE ---
    if gas.get('enforce_orientation', False):
        total_penalty += total_orientation_error_violation * gas['constraints']['orientation_weight']
    # --- END MODIFICATION 2 ---

    
    # 4. Environment boundary violation (hard constraint)
    if 'environment_bounds' in op and op['environment_bounds']:
        bounds = op['environment_bounds']
        for path in paths_xy:
            if np.any((path[:, 0] < bounds[0]) | (path[:, 0] > bounds[1]) |
                      (path[:, 1] < bounds[2]) | (path[:, 1] > bounds[3])):
                return float('inf')

    return total_penalty