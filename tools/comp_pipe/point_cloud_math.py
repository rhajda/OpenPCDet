import numpy as np
import math

def rotation_matrix(axis, theta):
    """
    Adapted from: https://stackoverflow.com/a/6802723/9621080
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate_pointcloud_y(point_cloud, y_axis_angle):
    rotation_correction = rotation_matrix(axis=[0, 1, 0], theta=y_axis_angle)
    return point_cloud @ rotation_correction.T