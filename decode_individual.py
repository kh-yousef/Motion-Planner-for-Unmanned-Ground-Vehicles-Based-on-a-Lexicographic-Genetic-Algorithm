import numpy as np
def decode_individual(op, chromosome):
    """
    Decodes a chromosome into path coordinates and orientations.
    This version STRICTLY cuts the path after the final segment, ensuring
    no overshoot in plots or metric calculations.
    """
    if chromosome is None or chromosome.size == 0:
        return [], []
        
    ntargets = chromosome.shape[0] - 1
    if ntargets < 0: # Handle single row chromosome for metrics
        ntargets = 1
        chromosome = np.vstack([chromosome, chromosome[-1,:]])

    n_nodes = op['n_nodes']
    
    # Indices for extra genes
    IDX_FINAL_SEG_IDX = n_nodes
    IDX_FINAL_ANGLE   = n_nodes + 1
    IDX_FINAL_LEN     = n_nodes + 2
    
    paths_xy = []
    orientations = []
    shared_lengths = chromosome[-1, :n_nodes]

    for i in range(ntargets):
        # Read cut-point and modification data from the chromosome
        final_segment_idx = int(round(chromosome[i, IDX_FINAL_SEG_IDX]))
        final_angle_deg   = chromosome[i, IDX_FINAL_ANGLE]
        final_len         = chromosome[i, IDX_FINAL_LEN]

        # Default to full path if index is not set
        if final_segment_idx <= 0:
            final_segment_idx = n_nodes

        # Build a temporary configuration that respects the cut point
        robot_configuration = np.zeros((n_nodes, 2)) # [angle, length]
        
        for j in range(n_nodes):
            if j < final_segment_idx:
                robot_configuration[j, 0] = chromosome[i, j]
                robot_configuration[j, 1] = shared_lengths[j]
            elif j == final_segment_idx:
                robot_configuration[j, 0] = final_angle_deg
                robot_configuration[j, 1] = final_len
            else:
                # Any segment AFTER the cut point has zero length
                robot_configuration[j, 0] = 0
                robot_configuration[j, 1] = 0
        
        # Determine the number of active nodes (segments with non-zero length)
        try:
            # Find the first segment with zero length
            active_nodes = np.where(robot_configuration[:, 1] == 0)[0][0]
        except IndexError:
            active_nodes = n_nodes
            
        # Perform forward kinematics on the CORRECTLY CUT path
        path_coords = np.zeros((active_nodes + 1, 2))
        orient_coords = np.zeros((active_nodes, 2))
        
        current_pos = np.array(op['home_base'][:2])
        heading_rad = op['home_base'][2]
        path_coords[0, :] = current_pos

        for j in range(active_nodes):
            angle_deg = robot_configuration[j, 0]
            length    = robot_configuration[j, 1]
            
            heading_rad += np.deg2rad(angle_deg)
            
            orient_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
            next_pos = current_pos + orient_vec * length
            
            path_coords[j + 1, :] = next_pos
            orient_coords[j, :] = orient_vec
            current_pos = next_pos

        paths_xy.append(path_coords)
        orientations.append(orient_coords)
        
    return paths_xy, orientations