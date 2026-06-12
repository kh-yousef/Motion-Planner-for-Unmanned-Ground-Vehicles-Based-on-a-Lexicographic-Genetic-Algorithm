def subtract_ranges(initial_ranges, forbidden_ranges):
    """
    Subtracts a list of forbidden ranges from an initial set of valid ranges.
    This is used to find the "safe openings" for angles.
    """
    if not forbidden_ranges:
        return initial_ranges
    
    # Sort forbidden ranges by their start value
    forbidden_ranges.sort(key=lambda x: x[0])
    
    # Merge overlapping forbidden ranges
    merged_forbidden = []
    if forbidden_ranges:
        current_forbidden = list(forbidden_ranges[0])
        for r_forbidden in forbidden_ranges[1:]:
            if r_forbidden[0] < current_forbidden[1]:
                current_forbidden[1] = max(current_forbidden[1], r_forbidden[1])
            else:
                merged_forbidden.append(current_forbidden)
                current_forbidden = list(r_forbidden)
        merged_forbidden.append(current_forbidden)
    
    # Subtract merged forbidden ranges from the current valid ranges
    current_ranges = list(initial_ranges)
    for frange in merged_forbidden:
        new_ranges = []
        for crange in current_ranges:
            # If there is no overlap, keep the current range as is
            if crange[1] < frange[0] or crange[0] > frange[1]:
                new_ranges.append(crange)
            else:
                # Left part of the current range is safe
                if crange[0] < frange[0]:
                    new_ranges.append([crange[0], frange[0]])
                # Right part of the current range is safe
                if crange[1] > frange[1]:
                    new_ranges.append([frange[1], crange[1]])
        current_ranges = new_ranges
        if not current_ranges:
            break
            
    return [r for r in current_ranges if r[0] < r[1]] # Return only valid ranges

def calculate_triangle_area(p1, p2, p3):
    return abs((p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[1]*p2[0] - p2[1]*p3[0] - p3[1]*p1[0]) / 2.0)