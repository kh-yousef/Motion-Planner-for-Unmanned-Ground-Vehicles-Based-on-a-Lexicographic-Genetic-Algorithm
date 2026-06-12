import numpy as np
def get_segment_circle_intersection_point(p1, p2, circle):
    """
    Find intersection point of line segment p1-p2 with circle (x, y, radius).
    Returns intersection point and boolean does_intersect.
    """
    center = np.array(circle[:2])
    radius = circle[2]
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)

    if a == 0:
        if np.linalg.norm(p1 - center) <= radius:
            return p1, True
        return None, False

    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None, False

    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    valid_points = []
    if 0 <= t1 <= 1:
        valid_points.append(p1 + t1 * d)
    if 0 <= t2 <= 1:
        valid_points.append(p1 + t2 * d)

    if len(valid_points) == 0:
        return None, False
    elif len(valid_points) == 1:
        return valid_points[0], True
    else:
        dist1 = np.linalg.norm(valid_points[0] - p1)
        dist2 = np.linalg.norm(valid_points[1] - p1)
        if dist1 < dist2:
            return valid_points[0], True
        else:
            return valid_points[1], True