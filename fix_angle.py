import numpy as np
def fix_angle(angle_deg):
    """
    Fix angle to be in [-180, 180] range.
    """
    angle = angle_deg - np.floor(angle_deg / 360) * 360
    if angle == -180:
        angle = 180
    if abs(angle) > 180:
        if angle > 0:
            angle = angle - 360
        else:
            angle = 360 + angle
    return angle
