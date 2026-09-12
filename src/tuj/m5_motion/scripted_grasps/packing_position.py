"""Opt-in nearby free-volume preference; actual release gates remain authoritative."""
from copy import copy
from itertools import product
import math
import numpy as np
from scipy.spatial.transform import Rotation

from .packing_occupancy import (
    packing_occupants, packing_overlap_preference, packing_translation_bounds,
    bounded_packing_xy,
)

POSITION_POLICY = 'NEAREST_CLEAR_VOLUME'
MAX_POSITION_SAMPLES = 4096


def restore_packing_position(g, points):
    """Carry one object's selected region-frame intent from TRANSPORT into PLACE."""
    from .transport import _is_region_place
    if (g.record.get('packing_metadata', {}).get('position_policy') != POSITION_POLICY
            or not _is_region_place(g.task)):
        return False
    retained = getattr(getattr(g, 'retention', None), 'packing_position', None)
    if (not retained or retained['object_id'] != g.object_id
            or retained['region_id'] != g.task.goal.target_region_id):
        return False
    rotation = Rotation.from_quat(retained['orientation_in_region_xyzw']).as_matrix()
    rotated = points @ rotation.T
    g.destination_rotation = g.T_WR[:3, :3] @ rotation
    g.preserve_destination_rotation = True
    g.half = np.ptp(points @ g.destination_rotation.T, axis=0) / 2.
    g.packing_bounds = (rotated.min(0), rotated.max(0))
    xy = bounded_packing_xy(g, retained['body_xy_in_region_m'])
    g.packing_body_xy = xy.copy()
    g.task.metadata['packing_position_selection'] = {**retained, 'status': 'RETAINED_FOR_PLACE'}
    return True


def select_clear_packing_position(g):
    """Keep the chosen orientation; move only to a nearer zero-overlap, IK-valid XY."""
    from .container_orientation import packing_transport_has_ik
    from .transport import _is_transport, transport_destination_center
    if (g.record.get('packing_metadata', {}).get('position_policy') != POSITION_POLICY
            or getattr(g, 'retention', None) is None or not _is_transport(g.task)):
        return
    # A new transport must not resurrect an intent selected for an earlier one.
    g.retention.packing_position = None
    if hasattr(g, 'packing_body_xy'):
        del g.packing_body_xy
    occupants = packing_occupants(g)
    if not occupants:
        return
    initial_score = packing_overlap_preference(g, occupants)
    evidence = {'policy': POSITION_POLICY, 'initial_score_m': initial_score,
                'basis': 'STATIC_VOLUME_PREFERENCE_ONLY', 'evaluated_xy_count': 1}
    g.task.metadata['packing_position_selection'] = evidence
    if initial_score == 0.:
        evidence['status'] = 'KEEP_CLEAR_PREFERRED_POSITION'
        return
    low, high = packing_translation_bounds(g)
    resolution = float(g.request.constraints.position_tolerance_m)
    body = transport_destination_center(g) - g.destination_rotation @ g.center_in_body
    preferred = (g.T_WR[:3, :3].T @ (body - g.T_WR[:3, 3]))[:2]
    counts = [max(2, 1 + math.ceil(float(high[i] - low[i]) / resolution)) for i in range(3)]
    # Check the budget before allocating an arbitrarily large grid. Including
    # preferred coordinates adds at most one sample to either horizontal axis.
    if ((counts[0] + 1) * (counts[1] + 1) + 1) * counts[2] > MAX_POSITION_SAMPLES:
        evidence.update(status='GRID_BUDGET_EXCEEDED_KEEP_EXISTING',
                        maximum_samples=MAX_POSITION_SAMPLES)
        return
    axes = [np.unique(np.r_[np.linspace(low[i], high[i], counts[i]),
                            np.clip(preferred[i], low[i], high[i])]) for i in range(2)]
    locations = sorted(product(*axes), key=lambda xy: (
        float(np.linalg.norm(np.asarray(xy) - preferred)), float(xy[0]), float(xy[1])))
    evidence.update(preferred_body_xy_in_region_m=preferred.tolist(),
                    maximum_samples=MAX_POSITION_SAMPLES)
    for xy in locations:
        if np.array_equal(xy, preferred):
            continue
        evidence['evaluated_xy_count'] += 1
        score = packing_overlap_preference(g, occupants, body_xy_in_region=xy)
        if score != 0.:
            continue
        probe = copy(g)
        probe.packing_body_xy = np.asarray(xy)
        if not packing_transport_has_ik(probe):
            continue
        g.packing_body_xy = probe.packing_body_xy.copy()
        retained = {
            'object_id': g.object_id, 'region_id': g.task.goal.target_region_id,
            'body_xy_in_region_m': g.packing_body_xy.tolist(),
            'orientation_in_region_xyzw': Rotation.from_matrix(
                g.T_WR[:3, :3].T @ g.destination_rotation).as_quat().tolist(),
        }
        g.retention.packing_position = retained
        evidence.update(**retained, status='NEAREST_CLEAR_REACHABLE_POSITION',
                        selected_score_m=score,
                        displacement_m=float(np.linalg.norm(g.packing_body_xy - preferred)))
        return
    evidence['status'] = 'NO_CLEAR_REACHABLE_POSITION_KEEP_EXISTING'
