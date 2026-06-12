import numpy as np
from decode_individual import *
from point_to_segment_distance import *

def calculate_fitness(op, gas, chromosome):
    """
    Calculates fitness. This version implements the conditional logic
    to switch between the original WMR strategy and the robust
    "soft robot" docking strategy.
    
    MODIFIED: Re-calculates IK fitness based on the *final* cut path
    to resolve the "stale score" optimization conflict.
    
    FIXED: The re-calculated IK fitness now *only* considers the final
    alignment segment, not the entire path, to allow scores to drop to 0.
    """
    # --- Define extra gene indices for clarity ---
    n_nodes = op['n_nodes']
    IDX_FINAL_SEG_IDX = n_nodes
    IDX_FINAL_ANGLE   = n_nodes + 1
    IDX_FINAL_LEN     = n_nodes + 2
    IDX_ALIGN_SEG_IDX = n_nodes + 3
    
    mod_chromosome = np.copy(chromosome)
    
    # Decode the raw, uncut chromosome to get an initial path
    temp_chrom = np.copy(chromosome)
    temp_chrom[:, IDX_FINAL_SEG_IDX:] = 0 # Use raw angles/lengths
    paths_xy, orientations = decode_individual(op, temp_chrom)
    
    ntargets = len(paths_xy)
    total_ik_fitness = 0
    total_node_count = 0
    total_undulation = 0
    total_path_len = 0
    total_orn_error = 0 # No longer used but kept for consistent return signature

    for i in range(ntargets):
        path_coords = paths_xy[i]
        target_pose = op['targets'][i]
        target_pos = target_pose[:2]
        
        # This is the 'ik_fitness' calculated on the RAW, UNCUT path
        # We will use this to find the *cut point*, but NOT for the final score
        uncut_ik_fitness = 0.0
        
        if gas['enforce_orientation']:
            # --- 'TRUE' MODE: Use the Soft Robot docking approach ---
            orientation_segment = op['end_points'][i]
            
            # 1. Calculate distance from each node to the FINITE segment
            all_distances_to_segment = np.array([
                point_to_segment_distance(pt, orientation_segment[0], orientation_segment[1])
                for pt in path_coords
            ])
            
            # 2. Find alignment point as the node closest to the ORIENTATION SEGMENT
            align_node_idx = np.argmin(all_distances_to_segment)
            
            if align_node_idx == 0 and len(path_coords) > 1:
                align_node_idx = 1
                
            # 3. Calculate (temporary) IK fitness based on distances up to the alignment point
            dist_vect = all_distances_to_segment[:align_node_idx + 1]
            weights = np.arange(1, align_node_idx + 2)
            weights[-1] *= 3  # Emphasize the alignment point
            
            # This is the "stale" fitness score, based on the *uncut* path
            uncut_ik_fitness = np.dot(dist_vect, weights) / np.sum(weights) if np.sum(weights) > 0 else 0
            
            # 4. The cut point is determined by the alignment point
            final_segment_idx = max(0, align_node_idx - 1)

        else:
                      # --- 'FALSE' MODE: Reach target, ignore orientation (USER REQUEST) ---
            # We do not care about the orientation line, so the IK fitness
            # contribution (related to orientation) is 0.
            uncut_ik_fitness = 0.0
            
            # We still need to find the "cut point" based on physical
            # reachability to ensure the path *ends* at the target.
            
            # 'align_node_idx' is not used for fitness, set to 0 as placeholder.
            # This value is stored in the chromosome but not used in this mode.
            align_node_idx = 0 
            
            # Find cut point using physical reachability.
            # Start search from the first segment (k=1).
            final_segment_idx = len(path_coords) - 2 # Default to second to last segment
            if final_segment_idx < 0: 
                final_segment_idx = 0

            for k in range(1, len(path_coords)): # Start from k=1
                dist_to_target_k = np.linalg.norm(path_coords[k] - target_pos)
                # Check length of the *previous* segment (the one leading to node k)
                original_segment_len = chromosome[-1, k-1] if k-1 < n_nodes else 0
                
                if dist_to_target_k < original_segment_len:
                    final_segment_idx = k - 1
                    break

        # --- 4. Recalculate Final Segment and Store Results (COMMON TO BOTH MODES) ---
        p_start_final_segment = path_coords[final_segment_idx]
        vec_to_target = target_pos - p_start_final_segment
        new_final_len = np.linalg.norm(vec_to_target)

        # Get orientation vector at the start of the final segment
        orient_vec_at_start = orientations[i][final_segment_idx -1] if final_segment_idx > 0 else np.array([np.cos(op['home_base'][2]), np.sin(op['home_base'][2])])

        global_angle_to_target_rad = np.arctan2(vec_to_target[1], vec_to_target[0])
        global_orient_at_start_rad = np.arctan2(orient_vec_at_start[1], orient_vec_at_start[0])
        
        new_relative_angle_rad = global_angle_to_target_rad - global_orient_at_start_rad
        new_final_angle_deg = np.degrees(np.arctan2(np.sin(new_relative_angle_rad), np.cos(new_relative_angle_rad)))

        # Update the chromosome with the new cut/modification data
        mod_chromosome[i, IDX_FINAL_SEG_IDX] = final_segment_idx
        mod_chromosome[i, IDX_FINAL_ANGLE]   = new_final_angle_deg
        mod_chromosome[i, IDX_FINAL_LEN]     = new_final_len
        mod_chromosome[i, IDX_ALIGN_SEG_IDX] = align_node_idx

        # --- 5. Recalculate Final Metrics using the CUT path ---
        # Decode the just-modified single target chromosome to get the final, true path
        single_target_chrom = np.vstack([mod_chromosome[i, :], mod_chromosome[-1, :]])
        final_paths_xy, _ = decode_individual(op, single_target_chrom)
        final_path_coords = final_paths_xy[0] # This is the REAL path.

        # --- START FIX ---
        # RE-CALCULATE IK FITNESS based on the *actual* final path
        # The original 'uncut_ik_fitness' was based on the uncut path and is now stale.
        # We now calculate the *true* IK fitness from the *final* path.
        
        if gas['enforce_orientation']:
            orientation_segment = op['end_points'][i]
            
            final_node_count = len(final_path_coords)
            
            if final_node_count > 1:
                # --- THIS IS THE FIX ---
                # We only measure the alignment of the *last two points*:
                # 1. The start of the final segment (final_path_coords[-2])
                # 2. The target itself (final_path_coords[-1])
                # This prevents large distances from the home_base from
                # polluting the score.
                last_two_points = final_path_coords[-2:]
                
                final_distances_to_segment = np.array([
                    point_to_segment_distance(pt, orientation_segment[0], orientation_segment[1])
                    for pt in last_two_points
                ])
                
                # Weight the start of the segment with 1, and the target with 3
                weights = np.array([1, 3])
                ik_fitness = np.dot(final_distances_to_segment, weights) / np.sum(weights) if np.sum(weights) > 0 else 0
                # --- END OF FIX ---
                
            elif final_node_count == 1:
                # Only one point (home base == target?), check its distance
                dist = point_to_segment_distance(final_path_coords[0], orientation_segment[0], orientation_segment[1])
                ik_fitness = dist
            else:
                # No path, no error
                ik_fitness = 0
        else:
            # If not enforcing orientation, the original (stale) IK is what we use
            ik_fitness = uncut_ik_fitness
            
        # --- END FIX ---


        # Calculate other metrics based on the final, cut path (as before)
        node_count = len(final_path_coords) - 1
        path_length = np.sum(np.linalg.norm(np.diff(final_path_coords, axis=0), axis=1))
        
        angles_rad = np.deg2rad(mod_chromosome[i, :node_count - 1])
        undulation = np.sum(np.diff(np.sign(angles_rad)) != 0) if len(angles_rad) > 1 else 0

        # Accumulate metrics
        total_ik_fitness += ik_fitness # Add the *newly calculated* true IK fitness
        total_node_count += node_count
        total_undulation += undulation
        total_path_len += path_length

    # Average metrics over all targets
    avg_ik_fitness = total_ik_fitness / ntargets
    avg_node_count = total_node_count / ntargets
    avg_undulation = total_undulation / ntargets
    avg_path_len = total_path_len / ntargets

    # Return the *true* ik_fitness, not the stale one
    return mod_chromosome, avg_ik_fitness, avg_node_count, avg_undulation, avg_path_len, total_orn_error