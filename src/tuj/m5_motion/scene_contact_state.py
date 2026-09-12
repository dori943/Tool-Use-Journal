"""Carry calibrated scene contact response across an EE topology replacement."""
from dataclasses import dataclass
import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class SceneGeomSolref:
    body_name: str
    geom_type: int
    solref: tuple[float, float]


def capture_scene_solref(env, model):
    roots = set()
    for robot in getattr(env, 'robots', ()):
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.root_body)
        if root < 0:
            raise ValueError('SCENE_CONTACT_ROBOT_ROOT_NOT_FOUND')
        roots.add(root)
    result = {}
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if not name:
            continue
        body = int(model.geom_bodyid[gid])
        ancestor = body
        while ancestor and ancestor not in roots:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor in roots:
            continue
        reference = tuple(float(x) for x in model.geom_solref[gid])
        if not np.isfinite(reference).all():
            raise ValueError('SCENE_CONTACT_NONFINITE_SOLREF')
        result[name] = SceneGeomSolref(model.body(body).name, int(model.geom_type[gid]), reference)
    return result


def restore_scene_solref(env, model, saved):
    current = capture_scene_solref(env, model)
    updates = []
    for name in saved.keys() & current.keys():
        source, target = saved[name], current[name]
        if source.body_name != target.body_name or source.geom_type != target.geom_type:
            raise ValueError(f'SCENE_CONTACT_GEOMETRY_IDENTITY_CHANGED: {name}')
        if not np.isfinite(source.solref).all():
            raise ValueError('SCENE_CONTACT_NONFINITE_SOLREF')
        updates.append((model.geom(name).id, source.solref))
    # Validate the complete common scene before changing the new model.
    for gid, reference in updates:
        model.geom_solref[gid] = reference
