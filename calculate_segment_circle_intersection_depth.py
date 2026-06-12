import numpy as np
def calculate_segment_circle_intersection_depth(op, p1, p2, circle):
    """
    Calculate the length of the intersection between line segment p1-p2 and circle (x,y,radius).
    """
    center = np.array(circle[0:2])
    obstacle_radius = circle[2]
    effective_radius = obstacle_radius + op['robot_radius']

    p1 = np.array(p1)
    p2 = np.array(p2)

    d = p2 - p1
    f = p1 - center

    a = np.dot(d, d)
    if a == 0:
        if np.linalg.norm(f) < effective_radius:
            return 0
        else:
            return 0

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - effective_radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return 0

    discriminant = np.sqrt(discriminant)

    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    overlap_start = max(0, min(t1, t2))
    overlap_end = min(1, max(t1, t2))

    if overlap_end > overlap_start:
        return (overlap_end - overlap_start) * np.linalg.norm(d)
    else:
        return 0
