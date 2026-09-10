"""Named-node coupling of a constitutive material to native MuJoCo dynamics.

Call prepare_forces after mj_step1 and before mj_step2. This component never
prescribes material positions during execution; only explicit reset/restore
operations may change nodal state.
"""
import numpy as np
import mujoco


def contact_endpoint_name(model, contact, side):
    """Resolve native contact endpoints without treating flex's -1 as a geom."""
    geom = int(contact.geom[side])
    if geom >= 0:
        return model.geom(geom).name
    flex = int(contact.flex[side])
    if flex >= 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex)
    raise ValueError("contact endpoint has neither geom nor flex identity")


def check_native_contact(model, data, first, second=None):
    def names(value):
        if value is None:
            return None
        if isinstance(value, str):
            return {value}
        if hasattr(value, "contact_geoms"):
            result = set(value.contact_geoms)
            if hasattr(value, "flex"):
                result.add(value.flex.get("name"))
            return result
        return set(value)
    left, right = names(first), names(second)
    for contact in data.contact[:data.ncon]:
        a, b = (contact_endpoint_name(model, contact, side) for side in (0, 1))
        if ((a in left and (right is None or b in right))
                or (b in left and (right is None or a in right))):
            return True
    return False


class FlexMaterialRuntime:
    def __init__(self, model, data, obj):
        self.model, self.data, self.obj = model, data, obj
        self.material = obj.new_material()
        self.bodies = np.array([model.body(obj.naming_prefix + f"node_{i}").id
                               for i in range(len(obj.reference_positions))])
        self.joints = [model.joint(obj.naming_prefix + f"node_{i}_{a}").id
                       for i in range(len(self.bodies)) for a in range(3)]
        self.qpos = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        self.root = model.body(obj.root_body).id
        self.mocap = int(model.body_mocapid[self.root])
        self.last_time = None

    def prepare_forces(self):
        positions = self.data.xpos[self.bodies].copy()
        now = float(self.data.time)
        if self.last_time is not None and now < self.last_time:
            raise ValueError("material time moved backwards without explicit restore")
        if now != self.last_time:
            self.material.update_plasticity(positions)
            self.last_time = now
        _, forces = self.material.energy_and_forces(positions)
        # Owned material nodes only; preserve robot and other external forces.
        self.data.xfrc_applied[self.bodies, :3] = forces

    def reset(self, position, quaternion_wxyz):
        self.data.mocap_pos[self.mocap] = position
        self.data.mocap_quat[self.mocap] = quaternion_wxyz
        self.data.qpos[self.qpos] = 0
        self.data.qvel[self.dofs] = 0
        self.data.xfrc_applied[self.bodies] = 0
        self.material = self.obj.new_material()
        self.last_time = None
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return {"material": self.material.state(),
                "qpos": self.data.qpos[self.qpos].tolist(),
                "qvel": self.data.qvel[self.dofs].tolist(),
                "position": self.data.mocap_pos[self.mocap].tolist(),
                "quaternion_wxyz": self.data.mocap_quat[self.mocap].tolist()}

    def restore(self, state):
        arrays = {key: np.asarray(state[key], dtype=float) for key in
                  ("qpos", "qvel", "position", "quaternion_wxyz")}
        for key, shape in (("qpos", (len(self.qpos),)), ("qvel", (len(self.dofs),)),
                           ("position", (3,)), ("quaternion_wxyz", (4,))):
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"invalid material {key}")
        if not np.isclose(np.linalg.norm(arrays["quaternion_wxyz"]), 1):
            raise ValueError("material root quaternion must be normalized")
        self.material.restore(state["material"])
        self.data.qpos[self.qpos] = arrays["qpos"]
        self.data.qvel[self.dofs] = arrays["qvel"]
        self.data.mocap_pos[self.mocap] = arrays["position"]
        self.data.mocap_quat[self.mocap] = arrays["quaternion_wxyz"]
        self.data.xfrc_applied[self.bodies] = 0
        self.last_time = None
        mujoco.mj_forward(self.model, self.data)

    def metrics(self):
        vertices = self.data.xpos[self.bodies]
        return {"bounds_min_m": vertices.min(axis=0).tolist(),
                "bounds_max_m": vertices.max(axis=0).tolist(),
                "height_m": float(np.ptp(vertices[:, 2])),
                "volume_ratio": float(self.material.volumes(vertices).sum()
                                      / self.material.reference_volumes.sum())}

    def geometry_record(self):
        rotation = self.data.xmat[self.root].reshape(3, 3)
        local = (self.data.xpos[self.bodies] - self.data.xpos[self.root]) @ rotation
        lower, upper = local.min(axis=0) - self.obj.radius, local.max(axis=0) + self.obj.radius
        center = (lower + upper) / 2
        return {"dimensions_m": (upper - lower).tolist(),
                "collision_enabled": True,
                "flex_name": self.obj.flex.get("name"),
                "collision_points_m": local.tolist(),
                "anchors": {"center": center.tolist(),
                            "top": [float(center[0]), float(center[1]), float(upper[2])],
                            "top_center": [float(center[0]), float(center[1]), float(upper[2])],
                            "bottom": [float(center[0]), float(center[1]), float(lower[2])]},
                "deformable_metrics": self.metrics()}
