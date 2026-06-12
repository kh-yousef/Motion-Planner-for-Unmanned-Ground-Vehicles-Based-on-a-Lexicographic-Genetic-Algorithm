import numpy as np
from calculate_single_orientation_segment import *

def precalculate_all_orientation_segments(op):
    """
    Pre-calculate orientation line segments for all targets as vectors for efficient reuse.
    Returns a list of 2x2 numpy arrays for each target, representing the start and end points of the segment.
    """
    n_targets = op['targets'].shape[0]
    all_segments = []

    for i in range(n_targets):
        # Pass 'op' to the helper function
        segment = calculate_single_orientation_segment(op, op['targets'][i])
        all_segments.append(segment)

    return all_segments

