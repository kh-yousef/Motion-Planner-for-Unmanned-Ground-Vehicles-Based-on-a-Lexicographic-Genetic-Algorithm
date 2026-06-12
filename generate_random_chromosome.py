import numpy as np
from get_random_angle_avoiding_obstacles_wmr import *

def generate_random_chromosome(op, gas):
    """
    Generate a single random chromosome encoding robot path segments.
    For angle genes, random values within angle domain or obstacle-avoidance aware sampling.
    For length genes, random values within length domain.
    """
    n_targets = op['targets'].shape[0]
    chrom = np.zeros((n_targets + 1, op['n_nodes'] + gas['extra_genes']))

    # Random lengths for segments shared across targets, stored in last row
    lengths = np.random.uniform(op['length_domain'][0], op['length_domain'][1], op['n_nodes'])
    chrom[-1, :op['n_nodes']] = lengths

    # Generate angles per target row
    for i in range(n_targets):
        current_pos = np.array(op['home_base'][:2])
        current_orientation_vec = np.array([np.cos(op['home_base'][2]), np.sin(op['home_base'][2])])

        for j in range(op['n_nodes']):
            length = lengths[j]

            if hasattr(gas, 'obstacle_avoidance') and gas.obstacle_avoidance:
                angle = get_random_angle_avoiding_obstacles_wmr(op, current_pos, current_orientation_vec, length, op['angle_domain'])
            else:
                angle = np.random.uniform(op['angle_domain'][0], op['angle_domain'][1])

            chrom[i, j] = angle

            # Update orientation vector applying rotation by angle
            alpha_rad = np.deg2rad(angle)
            rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                                        [np.sin(alpha_rad), np.cos(alpha_rad)]])
            new_orientation_vec = rotation_matrix.dot(current_orientation_vec)
            # Move current position forward by length along new orientation
            current_pos = current_pos + new_orientation_vec * length
            current_orientation_vec = new_orientation_vec

    return chrom