import numpy as np
from obstacle_avoidance_get_angle_wmr import *
def get_random_angle_avoiding_obstacles_wmr(op, current_pos, current_orn_vec, max_len, angle_domain):
    """Picks a random valid angle from the calculated safe ranges."""
    safe_ranges = obstacle_avoidance_get_angle_wmr(op, current_pos, current_orn_vec, max_len, angle_domain)
    
    if not safe_ranges:
        # No safe angle found, return a random angle from the original domain as a fallback
        return np.random.uniform(angle_domain[0], angle_domain[1])
        
    total_range_size = sum(r[1] - r[0] for r in safe_ranges)
    r = np.random.uniform(0, total_range_size)
    
    cumulative_size = 0
    for r_range in safe_ranges:
        range_size = r_range[1] - r_range[0]
        if r < cumulative_size + range_size:
            return r_range[0] + (r - cumulative_size)
        cumulative_size += range_size
        
    return safe_ranges[-1][1] # Fallback to the edge of the last safe range