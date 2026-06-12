import numpy as np
def find_nearby_obstacles(op, current_pos, max_len, obstacles):
    """Identifies obstacles that are close enough to be a potential threat."""
    nearby_indices = []
    if obstacles is None or len(obstacles) == 0:
        return nearby_indices
        
    for idx, obs in enumerate(obstacles):
        # Effective search distance is the max length of the new link plus the obstacle and robot radii
        search_dist = max_len + obs[2] + op['robot_radius']
        if np.linalg.norm(current_pos - obs[:2]) < search_dist:
            nearby_indices.append(idx)
    return nearby_indices