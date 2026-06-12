import numpy as np
def blend_values(p1_val, p2_val, alpha, bounds):
    """
    BLX-alpha operator to blend two gene values p1_val and p2_val with alpha.
    Ensures blended values are clipped within bounds.
    """
    min_val = min(p1_val, p2_val)
    max_val = max(p1_val, p2_val)
    range_ext = (max_val - min_val) * alpha
    lower = max(bounds[0], min_val - range_ext)
    upper = min(bounds[1], max_val + range_ext)

    c1 = np.random.uniform(lower, upper)
    c2 = np.random.uniform(lower, upper)
    return c1, c2
