import numpy as np
from fix_angle import *
from calculate_segment_length import *
from get_segment_circle_intersection_point import *

def calculate_single_orientation_segment(op, target_pose):
    """
    Calculate the orientation segment for a single target.
    Input target_pose is an array: [x, y, orientation_rad]
    """
    target_pos = target_pose[:2]
    target_orn_rad = target_pose[2]

    angle_deg = np.degrees(target_orn_rad) + 180
    angle_deg = fix_angle(angle_deg)

    u_vec = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])

    max_possible_len = op['n_nodes'] * op['length_domain'][1]
    seg_length = calculate_segment_length(target_pose, op['home_base'], max_possible_len)

    # Checking if environment bounds restrict segment length
    if 'environment_bounds' in op and op['environment_bounds'] is not None:
        bounds = op['environment_bounds']
        tx, ty = target_pos
        ux, uy = u_vec
        min_t_bounds = np.inf

        if ux != 0:
            t_left = (bounds[0] - tx) / ux
            t_right = (bounds[1] - tx) / ux
            if t_left > 0 and (ty + t_left * uy >= bounds[2]) and (ty + t_left * uy <= bounds[3]):
                min_t_bounds = min(min_t_bounds, t_left)
            if t_right > 0 and (ty + t_right * uy >= bounds[2]) and (ty + t_right * uy <= bounds[3]):
                min_t_bounds = min(min_t_bounds, t_right)

        if uy != 0:
            t_bottom = (bounds[2] - ty) / uy
            t_top = (bounds[3] - ty) / uy
            if t_bottom > 0 and (tx + t_bottom * ux >= bounds[0]) and (tx + t_bottom * ux <= bounds[1]):
                min_t_bounds = min(min_t_bounds, t_bottom)
            if t_top > 0 and (tx + t_top * ux >= bounds[0]) and (tx + t_top * ux <= bounds[1]):
                min_t_bounds = min(min_t_bounds, t_top)

        if min_t_bounds < np.inf:
            seg_length = min(seg_length, min_t_bounds)

    p_initial = target_pos + u_vec * seg_length
    closest_intersection_dist = np.inf
    p_final = p_initial

    if 'obstacles' in op and op['obstacles'] is not None and len(op['obstacles']) > 0:
        for obstacle in op['obstacles']:
            intersect_pt, does_intersect = get_segment_circle_intersection_point(target_pos, p_initial, obstacle)
            if does_intersect:
                dist = np.linalg.norm(intersect_pt - target_pos)
                if dist < closest_intersection_dist:
                    closest_intersection_dist = dist
                    p_final = intersect_pt

    return np.vstack([target_pos, p_final])