"""Necessary open-hand clearance for gravity release, not a release certificate."""
import math
import numpy as np


class OpenHandDropCorridor:
    def __init__(self, context, margin, position_tolerance):
        from .open_geometry import open_joint_positions
        from tuj.m5_motion.tool_use_journal import _geom_local_points
        c = context
        self.mj, self.model = c.mj, c.model
        self.margin = float(margin)
        tolerance = float(position_tolerance)
        if (not math.isfinite(self.margin) or self.margin < 0
                or not math.isfinite(tolerance) or tolerance <= 0):
            raise ValueError('RELEASE_CORRIDOR_INVALID_CLEARANCE')
        self.resolution = min(self.margin, tolerance) if self.margin > 0 else tolerance
        gravity = np.asarray(self.model.opt.gravity).copy()
        norm = float(np.linalg.norm(gravity))
        if not np.isfinite(gravity).all() or not math.isfinite(norm) or norm <= 0:
            raise ValueError('RELEASE_CORRIDOR_GRAVITY_REQUIRED')
        self.gravity = gravity / norm
        jid = int(self.model.body_jntadr[c.body_id])
        if (jid < 0 or self.model.jnt_type[jid] != self.mj.mjtJoint.mjJNT_FREE
                or self.model.body_parentid[c.body_id] != 0):
            raise ValueError('RELEASE_CORRIDOR_FREE_ROOT_OBJECT_REQUIRED')
        self.address = int(self.model.jnt_qposadr[jid])
        self.data = self.mj.MjData(self.model)
        self.data.qpos[:] = c.data.qpos
        for name, value in open_joint_positions(c).items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        self.mj.mj_fwdPosition(self.model, self.data)
        self.initial_position = self.data.qpos[self.address:self.address+3].copy()
        self.rotation = self.data.xmat[c.body_id].reshape(3, 3).copy()

        def enabled(ids):
            return [gid for gid in sorted(ids)
                    if self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]]

        self.hand = enabled(c.gripper_geoms)
        self.objects = enabled(c.object_geoms)

        def points(ids):
            if not ids:
                raise ValueError('RELEASE_CORRIDOR_COLLISION_GEOMETRY_REQUIRED')
            parts = []
            for gid in ids:
                local = _geom_local_points(self.model, gid)
                if local is None:
                    raise ValueError('RELEASE_CORRIDOR_UNSUPPORTED_COLLISION_GEOMETRY')
                parts.append(local @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid])
            result = np.concatenate(parts)
            if not np.isfinite(result).all():
                raise ValueError('RELEASE_CORRIDOR_NONFINITE_GEOMETRY')
            return result

        self.hand_points, self.object_points = points(self.hand), points(self.objects)

    def evaluate(self, destination_rotation):
        rotation = np.asarray(destination_rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0.)):
            raise ValueError('RELEASE_CORRIDOR_RIGID_ROTATION_REQUIRED')
        direction = self.rotation @ rotation.T @ self.gravity
        depth = max(0., float(np.max(self.hand_points @ direction)
                              - np.min(self.object_points @ direction) + self.margin))
        count = max(2, 1 + math.ceil(depth / self.resolution))
        interval = depth / (count-1)
        required = self.margin + interval / 2.
        cutoff = 2. * max(required, self.resolution)
        worst = {'distance_m': cutoff, 'drop_m': 0., 'geoms': None}
        closest = np.zeros(6)
        for drop in np.linspace(0., depth, count):
            self.data.qpos[self.address:self.address+3] = self.initial_position + direction * drop
            self.mj.mj_fwdPosition(self.model, self.data)
            for a in self.objects:
                for b in self.hand:
                    distance = float(self.mj.mj_geomDistance(self.model, self.data, a, b, cutoff, closest))
                    if not math.isfinite(distance):
                        raise ValueError('RELEASE_CORRIDOR_NONFINITE_DISTANCE')
                    if distance < worst['distance_m']:
                        worst = {'distance_m': distance, 'drop_m': float(drop),
                                 'geoms': [self.model.geom(a).name, self.model.geom(b).name]}
        return {'clear': worst['distance_m'] >= required, 'minimum': worst,
                'drop_direction_in_current_world': direction.tolist(),
                'depth_to_full_hand_separation_m': depth, 'sample_count': count,
                'sample_interval_m': interval, 'margin_m': self.margin,
                'required_sample_distance_m': required,
                'validation_scope': 'NECESSARY_OPEN_HAND_GRAVITY_DROP_CLEARANCE'}
