import numpy as np
from blend_values import *
from get_random_angle_avoiding_obstacles_wmr import *
def perform_blx_alpha_hybrid_avoidance(op, p1, p2, alpha):
    """
    Upgraded BLX-alpha crossover that uses proactive obstacle avoidance
    to generate intelligent, collision-aware offspring.
    """
    ntargets = p1.shape[0] - 1
    o1 = np.zeros_like(p1)
    o2 = np.zeros_like(p2)
    n_nodes = op['n_nodes']

    # Blend lengths first (they are shared)
    for j in range(n_nodes):
        o1[-1, j], o2[-1, j] = blend_values(p1[-1, j], p2[-1, j], alpha, op['length_domain'])

    for i in range(ntargets):
        # Initialize forward kinematics for each child
        child1_pos = op['home_base'][:2].astype(float)
        child1_orn = np.array([np.cos(op['home_base'][2]), np.sin(op['home_base'][2])])
        child2_pos = op['home_base'][:2].astype(float)
        child2_orn = np.array([np.cos(op['home_base'][2]), np.sin(op['home_base'][2])])
        
        for j in range(n_nodes):
            # Determine the blended range for the angle gene
            blended_angle1, blended_angle2 = blend_values(p1[i, j], p2[i, j], alpha, op['angle_domain'])
            angle_bound1 = [min(blended_angle1, p1[i, j]), max(blended_angle1, p1[i, j])]
            angle_bound2 = [min(blended_angle2, p2[i, j]), max(blended_angle2, p2[i, j])]

            # Get a random SAFE angle for each child from within their blended bounds
            temp_c1_angle = get_random_angle_avoiding_obstacles_wmr(op, child1_pos, child1_orn, o1[-1, j], angle_bound1)
            temp_c2_angle = get_random_angle_avoiding_obstacles_wmr(op, child2_pos, child2_orn, o2[-1, j], angle_bound2)
            
            o1[i, j] = temp_c1_angle
            o2[i, j] = temp_c2_angle
            
            # --- Update kinematics for Child 1 ---
            alpha1_rad = np.deg2rad(temp_c1_angle)
            R1 = np.array([[np.cos(alpha1_rad), -np.sin(alpha1_rad)], [np.sin(alpha1_rad), np.cos(alpha1_rad)]])
            new_child1_orn = R1.dot(child1_orn)
            child1_pos += new_child1_orn * o1[-1, j]
            child1_orn = new_child1_orn
            
            # --- Update kinematics for Child 2 ---
            alpha2_rad = np.deg2rad(temp_c2_angle)
            R2 = np.array([[np.cos(alpha2_rad), -np.sin(alpha2_rad)], [np.sin(alpha2_rad), np.cos(alpha2_rad)]])
            new_child2_orn = R2.dot(child2_orn)
            child2_pos += new_child2_orn * o2[-1, j]
            child2_orn = new_child2_orn
            
    return o1, o2