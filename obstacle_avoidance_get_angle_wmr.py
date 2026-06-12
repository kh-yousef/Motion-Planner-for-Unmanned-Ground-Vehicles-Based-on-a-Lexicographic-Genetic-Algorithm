import numpy as np
from find_nearby_obstacles import *
from subtract_ranges import *
def obstacle_avoidance_get_angle_wmr(op, current_pos, current_orn_vec, max_len, angle_domain):
    """
    Calculates all safe angular ranges by subtracting forbidden ranges.
    This is the core of the intelligent, proactive collision avoidance.
    """
    nearby_indices = find_nearby_obstacles(op, current_pos, max_len, op.get('obstacles'))
    if not nearby_indices:
        return [list(angle_domain)]

    # Rotate the frame so the robot's current orientation is along the x-axis
    angle_to_rotate = -np.arctan2(current_orn_vec[1], current_orn_vec[0])
    R = np.array([[np.cos(angle_to_rotate), -np.sin(angle_to_rotate)],
                  [np.sin(angle_to_rotate),  np.cos(angle_to_rotate)]])
                  
    forbidden_ranges = []
    for idx in nearby_indices:
        obs_center = op['obstacles'][idx][:2]
        obs_radius = op['obstacles'][idx][2]
        
        inflated_radius = obs_radius + op['robot_radius']
        relative_center = obs_center - current_pos
        obs_center_rotated = R.dot(relative_center)
        
        dist_to_center = np.linalg.norm(obs_center_rotated)
        
        if dist_to_center < inflated_radius:
            # The robot is already inside the inflated obstacle radius
            # All angles are forbidden in this case
            return [] 
            
        # Calculate the angle of the forbidden cone
        theta = np.arctan2(obs_center_rotated[1], obs_center_rotated[0])
        alpha_arg = np.clip(inflated_radius / dist_to_center, -1, 1)
        alpha = np.arcsin(alpha_arg)
        
        forbidden_ranges.append([np.degrees(theta - alpha), np.degrees(theta + alpha)])
        
    return subtract_ranges([list(angle_domain)], forbidden_ranges)