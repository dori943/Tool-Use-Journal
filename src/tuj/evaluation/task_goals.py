"""Whole-task predicates, independent of the planner's chosen subgoals.

Policy v1 is an explicit post-hoc operationalization of task_registry.py.
Snapshots must be measured runtime states, never M1/VLM or predicted states.
None means insufficient evidence; a known failed requirement still proves failure.
"""
from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tuj.m5_motion.flatten_contact import flattening_outcome
from tuj.m5_motion.push_to_region import target_resting_on_region
from tuj.m5_motion.container_surface import cover_geometry

POLICY_VERSION = "task-goals-v1"
FOOTPRINT_TOLERANCE_M = .0015
SUPPORT_TOLERANCE_M = .005
ALIGNMENT_TOLERANCE_DEG = 15.
TASKS = ("c1_1", "c2_1", "c3_1", "c1_2", "c2_2", "c3_2", "c4_1", "c4_2")


def task_from_world(world):
    name = world.get("metadata", {}).get("environment_name", "")
    return "_".join(name.lower().split("_")[:2])


def _pose(record):
    pose = record["pose"]
    p = np.asarray(pose["position_m"], dtype=float)
    q = np.asarray(pose["orientation_xyzw"], dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("missing or non-finite measured pose")
    return p, Rotation.from_quat(q).as_matrix()


def _local_points(record):
    points = np.asarray(record.get("collision_points_m"), dtype=float)
    if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 4 and np.isfinite(points).all():
        return points
    # Dimensions are a measured runtime bbox, not an assumed 1 cm default.
    dims = np.asarray(record.get("dimensions_m"), dtype=float)
    center = np.asarray(record.get("anchors", {}).get("center", [0., 0., 0.]), dtype=float)
    if dims.shape != (3,) or center.shape != (3,) or not np.isfinite(dims).all() or not np.isfinite(center).all() or np.any(dims <= 0):
        raise ValueError("missing measured collision geometry")
    return center + np.asarray(list(product((-1., 1.), repeat=3))) * dims / 2


def world_points(record):
    p, r = _pose(record)
    return p + np.einsum("ij,kj->ik", _local_points(record), r)


def _containment(objects, target, region, vertical=False):
    obj, base = objects[target], objects[region]
    p, r = _pose(base)
    points = np.einsum("ij,jk->ik", world_points(obj) - p, r)
    local = _local_points(base)
    lo, hi = local.min(axis=0), local.max(axis=0)
    packing = base.get("packing_metadata", {})
    if "interior_dimensions_m" in packing:
        size = np.asarray(packing["interior_dimensions_m"], dtype=float)
        center = np.asarray(packing["interior_center_m"], dtype=float)
        if size.shape != (3,) or center.shape != (3,) or not np.isfinite(size).all() or not np.isfinite(center).all() or np.any(size <= 0):
            raise ValueError("invalid measured interior geometry")
        lo, hi = center - size / 2, center + size / 2
    axes = 3 if vertical else 2
    excess = np.maximum(lo[:axes] - points[:, :axes].min(axis=0), points[:, :axes].max(axis=0) - hi[:axes])
    return bool(np.all(excess <= FOOTPRINT_TOLERANCE_M)), {
        "target": target, "region": region, "maximum_overhang_m": float(max(0., excess.max())),
        "tolerance_m": FOOTPRINT_TOLERANCE_M, "axes": axes,
    }


def _resting(objects, target, region):
    # Validate first: missing data must not become a default-size false/pass.
    for name in (target, region):
        world_points(objects[name])
    ok = target_resting_on_region(SimpleNamespace(objects=objects), target_id=target,
        region_id=region, horizontal_tolerance_m=FOOTPRINT_TOLERANCE_M,
        maximum_gap_m=SUPPORT_TOLERANCE_M, maximum_penetration_m=SUPPORT_TOLERANCE_M)
    return ok, {"target": target, "support": region, "gap_and_penetration_tolerance_m": SUPPORT_TOLERANCE_M}


def _support_chain(objects, target, region, peers):
    """Permit stacking on another packed item, but require a path to the tray."""
    pending = [(target, [target])]
    visited = set()
    while pending:
        name, path = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for base in [region, *peers]:
            if base in path:
                continue
            if _resting(objects, name, base)[0]:
                if base == region:
                    return True, {"support_path": path + [base], "tolerance_m": SUPPORT_TOLERANCE_M}
                pending.append((base, path + [base]))
    return False, {"target": target, "support": region, "reason": "no geometric support chain to region", "tolerance_m": SUPPORT_TOLERANCE_M}


def _released(world, names):
    robot = world["robot_state"]
    if "attached_object_id" not in robot or "held_tool_id" not in robot:
        raise ValueError("missing measured attachment state")
    held = {robot["attached_object_id"], robot["held_tool_id"]} - {None}
    held.update(world.get("metadata", {}).get("attached_object_transforms", {}))
    remaining = sorted(set(names) & held)
    return not remaining, {"still_held": remaining}


def evaluate_placement(world, target, region):
    """A released, contained object resting on its measured destination.

    Shared by task evaluation and live preservation checks; missing measured
    geometry is unknown and must never be interpreted as success.
    """
    checks = []
    for name, predicate in (
        ('contained', lambda: _containment(world['objects'], target, region)),
        ('supported', lambda: _resting(world['objects'], target, region)),
        ('released', lambda: _released(world, [target])),
    ):
        try:
            passed, evidence = predicate()
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
            passed, evidence = None, {'reason': f'insufficient evidence: {exc}'}
        checks.append({'name': name, 'passed': bool(passed) if passed is not None else None,
                       'evidence': evidence})
    values = [c['passed'] for c in checks]
    return {'target': target, 'region': region,
            'passed': False if False in values else (None if None in values else True),
            'checks': checks}


def _parallel(objects, a, b):
    axes = []
    for name in (a, b):
        points = _local_points(objects[name])
        axis = int(np.argmax(np.ptp(points, axis=0)))
        direction = _pose(objects[name])[1][:2, axis]
        if np.linalg.norm(direction) < 1e-8:
            return False, {"reason": "utensil long axis is vertical"}
        axes.append(direction / np.linalg.norm(direction))
    angle = float(np.degrees(np.arccos(np.clip(abs(axes[0] @ axes[1]), 0., 1.))))
    return angle <= ALIGNMENT_TOLERANCE_DEG, {"angle_deg": angle, "tolerance_deg": ALIGNMENT_TOLERANCE_DEG, "criterion_status": "USER_CONFIRMED_2026-10-25"}


def _separate(objects, a, b):
    hulls = []
    for name in (a, b):
        xy = world_points(objects[name])[:, :2]
        hulls.append(xy[ConvexHull(xy).vertices])
    for hull in hulls:
        edges = np.roll(hull, -1, axis=0) - hull
        for edge in edges:
            axis = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            p, q = [h @ axis for h in hulls]
            if min(p.max(), q.max()) - max(p.min(), q.min()) <= FOOTPRINT_TOLERANCE_M:
                return True, {"objects": [a, b], "overlap_tolerance_m": FOOTPRINT_TOLERANCE_M, "criterion_status": "USER_CONFIRMED_2026-10-25"}
    return False, {"objects": [a, b], "reason": "overlapping utensil footprints", "criterion_status": "USER_CONFIRMED_2026-10-25"}


def evaluate_task_goal(task, initial, final, *, history=None, cover_support=None):
    """Evaluate one terminal measured snapshot; no environment or API calls."""
    checks = []
    def check(name, function):
        try:
            value, evidence = function()
            value = bool(value) if value is not None else None
        except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
            value, evidence = None, {"reason": f"insufficient evidence: {type(exc).__name__}: {exc}"}
        checks.append({"name": name, "passed": value, "evidence": evidence})
    objects = (final or {}).get("objects", {})
    def inside(a, b, vertical=False):
        check(f"{a}.inside.{b}", lambda: _containment(objects, a, b, vertical))
    def resting(a, b):
        check(f"{a}.resting_on.{b}", lambda: _resting(objects, a, b))
    def release(names):
        check("required_objects_released", lambda: _released(final, names))
    def tray(names, region):
        for name in names:
            inside(name, region)
            check(f"{name}.supported_in.{region}", lambda name=name: _support_chain(objects, name, region, names))
        release(names)

    if task == "c1_1":
        names = [f"block_{i}" for i in range(12)]
        for name in names:
            inside(name, "collection_zone_visual")
            check(f"{name}.resting_on.collection_zone_visual",
                  lambda name=name: _support_chain(objects, name, "collection_zone_visual", names))
        release(names)
    elif task in ("c2_1", "c3_1"):
        names = ["apple", "bread", "mug", "plate", "spoon"]
        tray(names, "blue_tray")
        if task == "c2_1":
            def order():
                if history is None:
                    return None, {"reason": "missing execution history"}
                # Fixed C2_1 scene-observation camera: image-right = world +Y.
                expected = sorted(names, key=lambda n: _pose(initial["objects"][n])[0][1])
                sequences = {}
                for action in ("acquire", "place"):
                    events = [h for h in history if h.get("status") == "SUCCESS" and h.get("action_type", "").lower() == action]
                    if any(h.get("target_ids") is None for h in events):
                        return None, {"reason": "missing request for successful manipulation"}
                    sequences[action] = [n for h in events for n in h["target_ids"] if n in names]
                return all(seq == expected for seq in sequences.values()), {"expected": expected, "observed": sequences, "image_right_world_axis": "+Y"}
            check("left_to_right_execution_order", order)
    elif task == "c1_2":
        def flattened():
            record = objects["dough"]
            if not record.get("deformable_metrics", {}).get("tool_contact") or not record.get("flattening_configuration"):
                return None, {"reason": "missing measured deformation/contact/configuration"}
            ok, evidence = flattening_outcome(record)
            evidence["required_height_reduction_fraction"] = record["flattening_configuration"]["height_reduction_fraction"]
            return ok, evidence
        check("measured_unloaded_flattening", flattened)
    elif task == "c2_2":
        names = ["bread_a", "bread_b", "turkey_1", "cheese_1", "tomato_slice"]
        def sandwich():
            positions = {n: _pose(objects[n])[0] for n in names}
            plate = _pose(objects["serving_plate"])[0]
            order = sorted(names, key=lambda n: positions[n][2])
            ok = {order[0], order[-1]} == {"bread_a", "bread_b"} and all(np.linalg.norm(p[:2] - plate[:2]) < .09 and p[2] > plate[2] for p in positions.values())
            return ok, {"bottom_to_top": order, "plate_center_tolerance_m": .09}
        check("sandwich_composition_and_order", sandwich)
        def support():
            order = sorted(names, key=lambda n: _pose(objects[n])[0][2])
            pairs = list(zip(order, ["serving_plate", *order[:-1]]))
            outcomes = [_resting(objects, a, b)[0] for a, b in pairs]
            return all(outcomes), {"support_pairs": pairs, "passed": outcomes}
        check("sandwich_layers_supported", support)
        release(names)
    elif task == "c3_2":
        # Identical objects can swap trays: suffixes are identities, not goals.
        def assign(kind, regions):
            names = [f"{kind}_a", f"{kind}_b"]
            def score(order):
                evidence = [_containment(objects, n, region)[1] for n, region in zip(order, regions)]
                return sum(e["maximum_overhang_m"] for e in evidence)
            try:
                return min([names, names[::-1]], key=score)
            except (KeyError, ValueError, TypeError):
                return names  # Individual checks below will record missing evidence.
        regions = ["tray_a", "tray_b"]
        assignment = {kind: assign(kind, regions) for kind in ("plate", "mug", "fork", "spoon")}
        for kind in ("bread", "fruit"):
            assignment[kind] = assign(kind, assignment["plate"])
        for index, suffix in enumerate(("a", "b")):
            names = [assignment[kind][index] for kind in ("plate", "mug", "fork", "spoon")]
            tray(names, regions[index])
            for kind in ("bread", "fruit"):
                name = assignment[kind][index]
                inside(name, assignment["plate"][index])
                resting(name, assignment["plate"][index])
                release([name])
            a, b = assignment["fork"][index], assignment["spoon"][index]
            check(f"utensils_{suffix}.parallel", lambda a=a, b=b: _parallel(objects, a, b))
            check(f"utensils_{suffix}.separate", lambda a=a, b=b: _separate(objects, a, b))
    elif task == "c4_1":
        def extraction():
            old = initial["objects"]
            walls = ["appliance_left", "appliance_right"]
            separation = _pose(old[walls[1]])[0] - _pose(old[walls[0]])[0]
            across = separation[:2] / np.linalg.norm(separation[:2])
            axis = np.array([-across[1], across[0], 0.])
            before, after = world_points(old["card"]), world_points(objects["card"])
            travel = abs(float((_pose(objects["card"])[0] - _pose(old["card"])[0]) @ axis))
            cleared = []
            for state in (old, objects):
                span = np.concatenate([world_points(state[w]) @ axis for w in walls])
                extent = after @ axis
                cleared.append(bool(extent.max() <= span.min() + FOOTPRINT_TOLERANCE_M or extent.min() >= span.max() - FOOTPRINT_TOLERANCE_M))
            # A card knocked off the table must not count as extracted.
            bottom_change = float(after[:, 2].min() - before[:, 2].min())
            ok = travel >= .05 and all(cleared) and abs(bottom_change) <= SUPPORT_TOLERANCE_M
            return ok, {"channel_travel_m": travel, "minimum_travel_m": .05, "cleared_initial_and_final_flanks": cleared, "bottom_height_change_m": bottom_change, "height_tolerance_m": SUPPORT_TOLERANCE_M}
        check("card_extracted_from_gap", extraction)
        release(["card"])
    elif task == "c4_2":
        names = ["rolling_pin", "baguette", "whisk", "cereal", "milk"]
        for name in names:
            inside(name, "packing_box", True)
        def cover():
            result = cover_geometry(objects, "lid", "packing_box", SUPPORT_TOLERANCE_M, .05)
            return result["succeeded"], result
        check("lid_covers_opening", cover)
        def physical():
            if cover_support is None:
                return None, {"reason": "missing final lid placement physical settle report"}
            support = cover_support.get("surface_support", {})
            ok = cover_support.get("succeeded") is True and support.get("contact_count", 0) > 0 and support.get("upward_force_n", 0) > 0
            return ok, cover_support
        check("lid_physically_supported", physical)
        release([*names, "lid"])
    else:
        check("supported_task", lambda: (None, {"reason": f"unsupported task: {task}"}))
    values = [c["passed"] for c in checks]
    status = "FAILED" if False in values else "SATISFIED" if values and all(v is True for v in values) else "UNKNOWN"
    return {"schema": "task-goal-evaluation-v1", "policy_version": POLICY_VERSION,
        "task": task, "task_goal_status": status, "goal_success": {"SATISFIED": True, "FAILED": False}.get(status),
        "basis": "MEASURED_TERMINAL_RUNTIME_SNAPSHOT", "checks": checks,
        "failed_checks": [c["name"] for c in checks if c["passed"] is False],
        "unknown_checks": [c["name"] for c in checks if c["passed"] is None],
        "provisional_criteria": [],
        "neatness_criterion_status": "USER_CONFIRMED_2026-10-25" if task == "c3_2" else None}
