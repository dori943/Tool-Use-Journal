"""Transforms use metres, xyzw quaternions; T_AB maps B coordinates into A."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, quaternion_xyzw=None, rotation=None):
    result = np.eye(4)
    result[:3, 3] = np.asarray(position, dtype=float)
    result[:3, :3] = (np.asarray(rotation, dtype=float) if rotation is not None
                       else Rotation.from_quat(quaternion_xyzw).as_matrix())
    if not np.isfinite(result).all():
        raise ValueError("Non-finite pose")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-7):
        raise ValueError("Rotation must be orthonormal")
    if np.linalg.det(result[:3, :3]) < 0.999999:
        raise ValueError("Rotation must be right-handed")
    return result


def pose_dict(matrix):
    return {"position_m": matrix[:3, 3].tolist(),
            "orientation_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().tolist()}


def m1_center_to_world(center_mm, base_offset_mm):
    return (np.asarray(center_mm, float) + np.asarray(base_offset_mm, float)) / 1000.0


def inverse(matrix):
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result
