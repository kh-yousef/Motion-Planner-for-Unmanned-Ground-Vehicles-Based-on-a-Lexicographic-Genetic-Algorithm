import numpy as np
def point_to_segment_distance(pt, v, w):
    """
    Calculate the shortest distance from point pt to line segment vw.
    pt, v, w: np.array([x, y])
    """
    l2 = np.sum((w - v) ** 2)
    if l2 == 0:
        return np.linalg.norm(pt - v)
    t = max(0, min(1, np.dot(pt - v, w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(pt - projection)