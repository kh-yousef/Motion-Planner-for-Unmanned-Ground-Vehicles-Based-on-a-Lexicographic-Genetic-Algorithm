import numpy as np
from fix_angle import *

def calculate_segment_length(target_pose, home_base, max_length):
    """
    Calculate max segment length along orientation without crossing home base.
    """
    t = target_pose[:2]
    h = home_base[:2]
    target_angle_deg = np.degrees(target_pose[2])
    angle_deg = fix_angle(180 + target_angle_deg)

    h_translated = h - t
    angle_rot_rad = np.radians(-angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rot_rad), -np.sin(angle_rot_rad)],
        [np.sin(angle_rot_rad), np.cos(angle_rot_rad)]
    ])
    h_rotated = rot_matrix.dot(h_translated)
    h_rotated[0] = round(h_rotated[0], 3)

    if h_rotated[0] > 0:
        length = h_rotated[0]
    else:
        length = max_length

    return length
