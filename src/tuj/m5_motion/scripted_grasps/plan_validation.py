"""Audit planned endpoints with FK of the complete live simulator model."""
import numpy as np
from scipy.spatial.transform import Rotation


def audit_endpoints(context, plan, targets):
    c = context
    probe = c.mj.MjData(c.model)
    probe.qpos[:] = c.data.qpos
    rows = []
    for segment in plan.segments:
        key = segment.metadata.get('keyframe_id')
        if key not in targets:
            continue
        point = segment.waypoints[-1]
        probe.qpos[c.arm_ids] = point.joint_positions_rad
        c.mj.mj_kinematics(c.model, probe)
        goal = np.asarray(targets[key])
        position = probe.site_xpos[c.site_id].copy()
        rotation = probe.site_xmat[c.site_id].reshape(3, 3).copy()
        rows.append({'keyframe': key, 'position_error_m': float(np.linalg.norm(position-goal[:3, 3])),
            'orientation_error_deg': float(Rotation.from_matrix(goal[:3, :3].T@rotation).magnitude()*180/np.pi),
            'fk_position_world_m': position.tolist(), 'fk_rotation_world': rotation.tolist(),
            'joint_positions_rad': list(point.joint_positions_rad), 'fk_source': 'FULL_SIMULATOR_MODEL'})
    return rows
