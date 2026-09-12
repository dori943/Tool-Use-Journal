"""Pre-IK geometric gates for contact tool_act keyframes (e.g. sweep).

Rejects obviously unreachable or task-unrelated VLM poses before the compiler
spends an IK search. Scope is intentionally narrow:

- ``action_type == tool_act``
- ``contact.primitive`` in ``{sweep}``

Pick / place / transport and non-contact tool acts are no-ops.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tuj.m5_motion.geometry import (
    GeometryResolutionError,
    RelativePoseResolver,
    _anchor_local,
    _matrix_quaternion_xyzw,
    _pose_from_record,
)
from tuj.m5_motion.schema import (
    KeyframeType,
    MotionPlanRequest,
    Pose,
    RelativeKeyframeSpec,
)

# Contact primitives that must stay near task geometry (not free transport).
_VALIDATED_PRIMITIVES = frozenset({"sweep"})

# TCP within this ball of world origin is treated as a meaningless world-frame
# proposal (the documented VLM failure: frame_ref=world, anchor=origin).
_WORLD_ORIGIN_NEAR_M = 0.12

# Allow sweep contact near the support. Only reject poses that clearly punch
# through the table / floor (the original world-origin failure was z≈0).
_BELOW_SUPPORT_MARGIN_M = 0.10

# Sweep TRANSFER must stay near tool, targets, or collection region.
_TASK_RELEVANCE_RADIUS_M = 1.25

# When the live EEF pose is known, reject poses far beyond a conservative
# tabletop reach without touching IK tolerances.
_MAX_EEF_DISTANCE_M = 1.6

# Held-tool contact must keep TCP tool-z near the live grasp attitude.
# Cos(60°)≈0.5: near-horizontal proposals after height XY-retarget fail IK.
_MIN_HELD_TOOL_AXIS_DOT = 0.5

# Held-tool CONTACT_* TCP is the height where the tool face meets target tops
# (engagement), not the post-acquire lift height. After grasp the EEF is high;
# sweeping must lower so the held tool contacts targets without burying the
# tool into the support.
_CONTACT_BELOW_ENGAGEMENT_SLACK_M = 0.05
_CONTACT_ABOVE_ENGAGEMENT_SLACK_M = 0.08
# Keep the held-tool underside clear of the support by this margin.
_CONTACT_TOOL_ABOVE_SUPPORT_CLEARANCE_M = 0.005
# Shallow soft-contact press into sweep targets (not a zero-penetration kiss).
# Deep press tunnels free bodies through thin held tools; keep this small.
# Matches the physical C1 rim-push profile (~1 mm).
_CONTACT_SWEEP_PRESS_M = 0.0015
# Never press more than this fraction of held-tool thickness (anti-tunnel).
_CONTACT_SWEEP_PRESS_THICKNESS_FRAC = 0.15
# When sweeping targets that rest on the support, allow the tool underside this
# close to the bare table while clamping max press.
_CONTACT_SWEEP_SUPPORT_CLEARANCE_M = 0.001
# Held-tool collision cloud is treated as hollow when few points lie inside this
# fraction of the cloud's max radius (live C1 plate: empty disk r<~42 mm).
_HOLLOW_INNER_RADIUS_FRAC = 0.45
_HOLLOW_MIN_INNER_POINT_FRAC = 0.05
_HOLLOW_RIM_Z_BAND_M = 0.002
# Side-push contact height for hollow tools: fraction from support→target top.
_HOLLOW_RIM_CONTACT_HEIGHT_FRAC = 0.50
# Fallback band width when only the support surface is known.
_CONTACT_MAX_ABOVE_SUPPORT_M = 0.18
# If CONTACT_START is farther than this from the target-cluster centroid, snap
# it onto the centroid so the plate covers the group before the lateral push.
_CONTACT_START_CENTROID_SNAP_M = 0.04
_HOVER_MIN_ABOVE_SUPPORT_M = 0.06
_MIN_SWEEP_KEYFRAMES = 5
# Live eef_z - held-tool-bottom must lie in this range to be trusted.
_HELD_TOOL_BELOW_TCP_MAX_M = 0.45
# Minimum horizontal CONTACT_START → CONTACT_END travel. Zero-length sweeps
# (VLM collapses all CONTACT_* onto one block top) cannot move targets.
_MIN_SWEEP_LATERAL_M = 0.08
# Require this fraction of the start→region distance as forward progress when a
# goal region is known (generic; not task-id specific).
_MIN_SWEEP_REGION_PROGRESS_FRAC = 0.35

_CONTACT_PHASE_TYPES = frozenset(
    {
        KeyframeType.CONTACT_START,
        KeyframeType.CONTACT_SWEEP,
        KeyframeType.CONTACT_END,
    }
)
_HOVER_PHASE_TYPES = frozenset(
    {
        KeyframeType.PRE_CONTACT,
        KeyframeType.RETREAT,
    }
)

# Keyframe types that carry the contact path (not acquire/release machinery).
_CONTACT_PATH_TYPES = frozenset(
    {
        KeyframeType.TRANSFER,
        KeyframeType.CUSTOM,
        KeyframeType.PRE_GRASP,
        KeyframeType.GRASP,
        KeyframeType.LIFT,
        KeyframeType.PRE_PLACE,
        KeyframeType.PLACE,
        KeyframeType.RETREAT,
        KeyframeType.PRE_CONTACT,
        KeyframeType.CONTACT_START,
        KeyframeType.CONTACT_SWEEP,
        KeyframeType.CONTACT_END,
    }
)


class ContactKeyframeGeometryError(ValueError):
    """A contact tool_act keyframe fails geometric validation before IK."""


def is_tool_act_contact_geometry_scope(request: MotionPlanRequest) -> bool:
    """Whether contact keyframe geometry validation applies to this request."""

    action = str(request.task.action_type or "").strip().lower()
    if action != "tool_act":
        return False
    contact = request.task.contact
    if contact is None:
        return False
    primitive = str(contact.primitive or "").strip().lower()
    return primitive in _VALIDATED_PRIMITIVES


def validate_resolved_contact_keyframe(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    pose: Pose,
) -> None:
    """Raise ``ContactKeyframeGeometryError`` when ``pose`` is clearly invalid."""

    if not is_tool_act_contact_geometry_scope(request):
        return
    if keyframe.keyframe_type not in _CONTACT_PATH_TYPES:
        return

    position = np.asarray(pose.position_m, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: non-finite resolved TCP position"
        )

    _reject_world_origin_pose(keyframe, position)
    _reject_below_support(request, keyframe, position)
    _reject_contact_height_band(request, keyframe, position)
    _reject_inverted_held_tool_orientation(request, keyframe, pose)
    _reject_outside_workspace(request, keyframe, position)
    _reject_task_unrelated(request, keyframe, position)


def validate_sweep_keyframe_strategy(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    *,
    resolver: RelativePoseResolver | None = None,
) -> None:
    """Reject trivial 2-point transfers and enforce a contact-sweep sequence."""

    if not is_tool_act_contact_geometry_scope(request):
        return
    if not keyframes:
        raise ContactKeyframeGeometryError("sweep strategy has no keyframes")

    kinds = [keyframe.keyframe_type for keyframe in keyframes]
    if _is_trivial_two_point_transfer(kinds, keyframes):
        raise ContactKeyframeGeometryError(
            "sweep rejects trivial two-point TRANSFER "
            "(target → region); require PRE_CONTACT → CONTACT_START → "
            "CONTACT_SWEEP+ → CONTACT_END → RETREAT"
        )
    if len(keyframes) < _MIN_SWEEP_KEYFRAMES:
        raise ContactKeyframeGeometryError(
            f"sweep strategy requires at least {_MIN_SWEEP_KEYFRAMES} keyframes "
            f"(got {len(keyframes)}); use PRE_CONTACT → CONTACT_START → "
            "CONTACT_SWEEP → CONTACT_END → RETREAT"
        )
    _require_ordered_sweep_phases(kinds)

    active_resolver = resolver or RelativePoseResolver(request.world)
    support_z = _support_surface_z_m(request)
    contact_heights: list[float] = []
    for keyframe in keyframes:
        pose = active_resolver.resolve(keyframe)
        validate_resolved_contact_keyframe(request, keyframe, pose)
        if keyframe.keyframe_type in _CONTACT_PHASE_TYPES:
            contact_heights.append(float(pose.position_m[2]))

    if support_z is not None and contact_heights:
        max_contact = max(contact_heights)
        hover_floor = max(max_contact, support_z + _HOVER_MIN_ABOVE_SUPPORT_M)
        eef = request.world.robot_state.eef_pose
        if eef is not None and (
            request.world.robot_state.held_tool_id
            or request.world.robot_state.attached_object_id
        ):
            hover_floor = max(hover_floor, float(eef.position_m[2]) - 1e-6)
        for keyframe in keyframes:
            if keyframe.keyframe_type not in _HOVER_PHASE_TYPES:
                continue
            hover_z = float(active_resolver.resolve(keyframe).position_m[2])
            if hover_z < hover_floor - 1e-6:
                raise ContactKeyframeGeometryError(
                    f"{keyframe.keyframe_id}: {keyframe.keyframe_type.value} "
                    f"z={hover_z:.3f}m must stay at/above the contact working "
                    f"height (floor={hover_floor:.3f}m)"
                )
    _reject_degenerate_sweep_lateral(request, keyframes, active_resolver)


def canonicalize_held_tool_axis_for_contact(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    *,
    resolver: RelativePoseResolver | None = None,
) -> RelativeKeyframeSpec:
    """Flip ``tool_axis_to_align`` when it would invert the live held-tool TCP.

    VLMs frequently emit ``+z`` for top-down approaches even after a vacuum
    acquire that leaves TCP ``+z`` pointing down. Rewriting the symbolic axis
    preserves the proposed frame/anchor/offset while restoring a grasp-compatible
    attitude so candidates are not mass-rejected.
    """

    if not is_tool_act_contact_geometry_scope(request):
        return keyframe
    if keyframe.keyframe_type not in _CONTACT_PATH_TYPES:
        return keyframe
    if not (
        request.world.robot_state.held_tool_id
        or request.world.robot_state.attached_object_id
    ):
        return keyframe
    eef = request.world.robot_state.eef_pose
    if eef is None:
        return keyframe
    live_z = _pose_tool_z_world(eef.orientation_xyzw)
    if live_z is None:
        return keyframe

    active_resolver = resolver or RelativePoseResolver(request.world)
    try:
        current_pose = active_resolver.resolve(keyframe)
    except GeometryResolutionError:
        return keyframe
    current_z = _pose_tool_z_world(current_pose.orientation_xyzw)
    if current_z is None:
        return keyframe
    current_dot = float(np.dot(live_z, current_z))
    if current_dot >= _MIN_HELD_TOOL_AXIS_DOT:
        return keyframe

    flipped_axis = "+z" if keyframe.tool_axis_to_align == "-z" else "-z"
    candidate = keyframe.model_copy(
        update={
            "tool_axis_to_align": flipped_axis,
            "metadata": {
                **keyframe.metadata,
                "held_tool_axis_canonicalized": True,
                "held_tool_axis_from": keyframe.tool_axis_to_align,
            },
        }
    )
    try:
        flipped_pose = active_resolver.resolve(candidate)
    except GeometryResolutionError:
        return keyframe
    flipped_z = _pose_tool_z_world(flipped_pose.orientation_xyzw)
    if flipped_z is None:
        return keyframe
    flipped_dot = float(np.dot(live_z, flipped_z))
    # Only rewrite when the flip clearly restores alignment; horizontal
    # approaches often keep both dots near zero and should not be flipped.
    if flipped_dot < _MIN_HELD_TOOL_AXIS_DOT or flipped_dot <= current_dot + 1e-6:
        return keyframe
    return candidate


def canonicalize_held_tool_orientation_for_contact(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    *,
    resolver: RelativePoseResolver | None = None,
) -> RelativeKeyframeSpec:
    """Lock held-tool contact TCP attitude to the live grasp orientation.

    Height XY-retarget can rewrite ``approach_axis`` to a near-horizontal delta
    and freeze ``packing_orientation_xyzw`` from that attitude. Full-pose IK then
    fails even when the XY is reachable. After acquire, vac/gripper sweeps keep
    tool-z and only apply ``roll_rad`` about that axis.
    """

    del resolver  # API symmetry with sibling canonicalize helpers
    if not is_tool_act_contact_geometry_scope(request):
        return keyframe
    if keyframe.keyframe_type not in _CONTACT_PATH_TYPES:
        return keyframe
    packing = _live_held_tool_packing_xyzw(
        request, roll_rad=float(keyframe.roll_rad or 0.0)
    )
    if packing is None:
        return keyframe
    existing = keyframe.metadata.get("packing_orientation_xyzw")
    if (
        isinstance(existing, Sequence)
        and not isinstance(existing, (str, bytes))
        and len(existing) == 4
        and np.allclose(
            np.asarray(existing, dtype=float),
            np.asarray(packing, dtype=float),
            atol=1e-6,
        )
    ):
        return keyframe
    return keyframe.model_copy(
        update={
            "metadata": {
                **keyframe.metadata,
                "packing_orientation_xyzw": packing,
                "held_tool_orientation_locked": True,
            },
        }
    )


def contact_keyframe_geometry_errors(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    *,
    resolver: RelativePoseResolver | None = None,
) -> list[str]:
    """Return human-readable geometry failures for a strategy's keyframes."""

    if not is_tool_act_contact_geometry_scope(request):
        return []
    active_resolver = resolver or RelativePoseResolver(request.world)
    errors: list[str] = []
    for keyframe in keyframes:
        try:
            pose = active_resolver.resolve(keyframe)
            validate_resolved_contact_keyframe(request, keyframe, pose)
        except ContactKeyframeGeometryError as error:
            errors.append(str(error))
    return errors


def _reject_world_origin_pose(
    keyframe: RelativeKeyframeSpec, position: np.ndarray
) -> None:
    frame = str(keyframe.frame_ref or "").strip().lower()
    anchor = str(keyframe.anchor or "").strip().lower()
    distance = float(np.linalg.norm(position))
    near_origin = distance <= _WORLD_ORIGIN_NEAR_M
    symbolic_world_origin = frame == "world" and anchor in {"origin", "center"}
    if near_origin and (symbolic_world_origin or distance <= 1e-6):
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: resolved TCP near world origin "
            f"(frame_ref={keyframe.frame_ref!r}, anchor={keyframe.anchor!r}, "
            f"position_m={_fmt(position)}, ||p||={distance:.3f}m); "
            "use tool/target/region frames for contact tool_act"
        )


def _reject_below_support(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    position: np.ndarray,
) -> None:
    support_z = _support_surface_z_m(request)
    if support_z is None:
        return
    if float(position[2]) < support_z - _BELOW_SUPPORT_MARGIN_M:
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: resolved TCP z={float(position[2]):.3f}m "
            f"is below support surface z={support_z:.3f}m "
            f"(margin={_BELOW_SUPPORT_MARGIN_M}m)"
        )


def _reject_contact_height_band(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    position: np.ndarray,
) -> None:
    """Keep CONTACT_* TCP near held-tool ↔ target engagement height.

    Anchoring CONTACT_* on the post-grasp EEF leaves the tool hovering above
    targets. Anchoring on bare support with zero tool clearance embeds the
    tool into the table. Prefer target-top + held-tool-below-TCP.
    """

    if keyframe.keyframe_type not in _CONTACT_PHASE_TYPES:
        return
    limits = _contact_tcp_height_limits_m(request)
    if limits is None:
        return
    floor_z, ceiling_z = limits
    z_m = float(position[2])
    if z_m < floor_z or z_m > ceiling_z:
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: contact-phase TCP z={z_m:.3f}m is outside "
            f"the held-tool contact band [{floor_z:.3f}, {ceiling_z:.3f}]m; "
            "keep CONTACT_* near target engagement height (target top + held "
            "tool extent below TCP) so the tool face meets targets without "
            "embedding into the support"
        )


def _held_tool_below_tcp_m(request: MotionPlanRequest) -> float:
    """Vertical extent from live TCP down to the held-tool underside."""

    eef = request.world.robot_state.eef_pose
    if eef is not None:
        eef_z = float(eef.position_m[2])
        if math.isfinite(eef_z):
            for object_id in sorted(_held_object_ids(request)):
                bottom = _object_bottom_z(request.world.objects.get(object_id))
                if bottom is None:
                    continue
                below = eef_z - float(bottom)
                if 0.0 <= below <= _HELD_TOOL_BELOW_TCP_MAX_M:
                    return below
    return _held_tool_half_height_m(request)


def _sweep_target_top_z_m(request: MotionPlanRequest) -> float | None:
    held_ids = _held_object_ids(request)
    tops: list[float] = []
    for target_id in request.task.target_ids:
        if not isinstance(target_id, str) or not target_id.strip():
            continue
        object_id = target_id.strip()
        if object_id in held_ids:
            continue
        top = _object_top_z(request.world.objects.get(object_id))
        if top is not None:
            tops.append(top)
    if not tops:
        return None
    return max(tops)


def _held_tool_thickness_m(request: MotionPlanRequest) -> float | None:
    """Vertical thickness of the held tool from world snapshot dimensions."""

    for object_id in sorted(_held_object_ids(request)):
        record = request.world.objects.get(object_id)
        if not isinstance(record, Mapping):
            continue
        dims = _finite_vector(record.get("dimensions_m"), 3)
        if dims is not None and float(dims[2]) > 0.0:
            return float(dims[2])
        top = _object_top_z(record)
        bottom = _object_bottom_z(record)
        if top is not None and bottom is not None and top > bottom:
            return float(top - bottom)
    return None


def _held_tool_collision_points_body_m(
    request: MotionPlanRequest,
) -> np.ndarray | None:
    for object_id in sorted(_held_object_ids(request)):
        record = request.world.objects.get(object_id)
        if not isinstance(record, Mapping):
            continue
        points = record.get("collision_points_m")
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            continue
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 8:
            continue
        if not np.all(np.isfinite(array)):
            continue
        return array
    return None


def _held_tool_hollow_rim_inner_radius_m(
    request: MotionPlanRequest,
) -> float | None:
    """Inner rim radius for a hollow held-tool underside, else ``None``.

    Uses body-frame ``collision_points_m``. Flat slabs that fill the AABB disk
    return ``None`` so sweep keeps AABB top-press engagement.
    """

    points = _held_tool_collision_points_body_m(request)
    if points is None:
        return None
    radii = np.linalg.norm(points[:, :2], axis=1)
    r_max = float(np.max(radii))
    if not math.isfinite(r_max) or r_max < 1e-3:
        return None
    inner = float(_HOLLOW_INNER_RADIUS_FRAC) * r_max
    inner_count = int(np.count_nonzero(radii <= inner + 1e-12))
    if inner_count > max(3, int(_HOLLOW_MIN_INNER_POINT_FRAC * len(points))):
        return None
    z_min = float(np.min(points[:, 2]))
    rim = points[points[:, 2] <= z_min + float(_HOLLOW_RIM_Z_BAND_M)]
    if rim.shape[0] >= 4:
        return float(np.min(np.linalg.norm(rim[:, :2], axis=1)))
    return float(np.min(radii))


def _sweep_engagement_press_m(request: MotionPlanRequest) -> float:
    press = float(_CONTACT_SWEEP_PRESS_M)
    thickness = _held_tool_thickness_m(request)
    if thickness is not None and thickness > 0.0:
        press = min(
            press, float(thickness) * float(_CONTACT_SWEEP_PRESS_THICKNESS_FRAC)
        )
    return max(0.0, press)


def _contact_engagement_tcp_z_m(request: MotionPlanRequest) -> float | None:
    """TCP z for held-tool sweep engagement.

    Flat tools: shallow underside press into target tops.
    Hollow dishes (empty collision disk): put the rim underside near mid-target
    height so the rim wall side-pushes instead of hovering a bowl over tops.
    """

    tool_below = _held_tool_below_tcp_m(request)
    target_top = _sweep_target_top_z_m(request)
    support_z = _support_surface_z_m(request)
    press = _sweep_engagement_press_m(request)
    rim_r = _held_tool_hollow_rim_inner_radius_m(request)
    if (
        target_top is not None
        and support_z is not None
        and rim_r is not None
        and rim_r > 1e-4
    ):
        mid_z = float(support_z) + float(_HOLLOW_RIM_CONTACT_HEIGHT_FRAC) * (
            float(target_top) - float(support_z)
        )
        underside = mid_z - press
        underside = max(
            underside, float(support_z) + float(_CONTACT_SWEEP_SUPPORT_CLEARANCE_M)
        )
        return float(underside + tool_below)
    if target_top is not None:
        if support_z is not None:
            max_press = max(
                0.0,
                float(target_top)
                - float(support_z)
                - float(_CONTACT_SWEEP_SUPPORT_CLEARANCE_M),
            )
            press = min(press, max_press)
        return float(target_top + tool_below - press)
    if support_z is None:
        return None
    return float(
        support_z + tool_below + _CONTACT_TOOL_ABOVE_SUPPORT_CLEARANCE_M
    )


def _contact_tcp_height_limits_m(
    request: MotionPlanRequest,
) -> tuple[float, float] | None:
    tool_below = _held_tool_below_tcp_m(request)
    support_z = _support_surface_z_m(request)
    engagement = _contact_engagement_tcp_z_m(request)
    if engagement is None and support_z is None:
        return None
    if engagement is None:
        assert support_z is not None
        engagement = (
            support_z + tool_below + _CONTACT_TOOL_ABOVE_SUPPORT_CLEARANCE_M
        )
    floor_z = engagement - _CONTACT_BELOW_ENGAGEMENT_SLACK_M
    ceiling_z = engagement + _CONTACT_ABOVE_ENGAGEMENT_SLACK_M
    if support_z is not None:
        # Bare-support clearance keeps tools off the table. With sweep targets
        # present, allow a deeper floor so engagement press is not lifted back
        # into a zero-penetration kiss.
        clearance = (
            _CONTACT_SWEEP_SUPPORT_CLEARANCE_M
            if _sweep_target_top_z_m(request) is not None
            else _CONTACT_TOOL_ABOVE_SUPPORT_CLEARANCE_M
        )
        support_floor = support_z + tool_below + clearance
        floor_z = max(floor_z, support_floor)
        ceiling_z = max(ceiling_z, support_floor + 0.02)
    return floor_z, ceiling_z


def _contact_target_z_m(
    request: MotionPlanRequest, floor_z: float, ceiling_z: float
) -> float:
    engagement = _contact_engagement_tcp_z_m(request)
    if engagement is not None and math.isfinite(engagement):
        return float(min(max(engagement, floor_z), ceiling_z))
    return float(floor_z)


def _frame_anchor_world(
    resolver: RelativePoseResolver, keyframe: RelativeKeyframeSpec
) -> tuple[np.ndarray, np.ndarray]:
    record, is_rack = resolver._record(keyframe.frame_ref)
    frame_position, frame_rotation = _pose_from_record(record, rack=is_rack)
    anchor_world = frame_position + frame_rotation @ _anchor_local(
        record, keyframe.anchor
    )
    return anchor_world, frame_rotation


def canonicalize_contact_tcp_height(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    *,
    resolver: RelativePoseResolver | None = None,
) -> RelativeKeyframeSpec:
    """Move CONTACT_* / PRE_CONTACT / RETREAT TCP onto the sweep working height.

    CONTACT_* target the held-tool engagement plane (target top + tool-below-TCP).
    PRE_CONTACT / RETREAT stay at/above the live EEF (or that engagement plane).
    VLMs often keep CONTACT_* at post-grasp lift height (no contact) or drop them
    onto bare support (table collision). Retarget z while preserving XY when the
    approach is horizontal; allow negative offsets for downward approaches.
    """

    if not is_tool_act_contact_geometry_scope(request):
        return keyframe
    height_types = _CONTACT_PHASE_TYPES | _HOVER_PHASE_TYPES
    if keyframe.keyframe_type not in height_types:
        return keyframe

    floor_z: float
    ceiling_z: float | None
    target_z: float
    if keyframe.keyframe_type in _CONTACT_PHASE_TYPES:
        limits = _contact_tcp_height_limits_m(request)
        if limits is None:
            return keyframe
        floor_z, ceiling_z = limits
        target_z = _contact_target_z_m(request, floor_z, ceiling_z)
    else:
        hover_floor = _hover_working_floor_z_m(request)
        if hover_floor is None:
            return keyframe
        floor_z = hover_floor
        ceiling_z = None
        target_z = hover_floor

    active_resolver = resolver or RelativePoseResolver(request.world)
    try:
        pose = active_resolver.resolve(keyframe)
        anchor_world, frame_rotation = _frame_anchor_world(active_resolver, keyframe)
    except GeometryResolutionError:
        return keyframe
    z_m = float(pose.position_m[2])
    if abs(z_m - target_z) <= 1e-4 and (
        floor_z - 1e-6 <= z_m <= (ceiling_z + 1e-6 if ceiling_z is not None else z_m)
    ):
        return keyframe
    in_band = floor_z - 1e-6 <= z_m and (
        ceiling_z is None or z_m <= ceiling_z + 1e-6
    )
    # Hover phases only lift; CONTACT_* both lift and lower into engagement.
    if keyframe.keyframe_type in _HOVER_PHASE_TYPES:
        if in_band or z_m >= floor_z - 1e-6:
            return keyframe
    elif in_band:
        return keyframe

    candidate = _retarget_tcp_height(
        keyframe,
        pose=pose,
        anchor_world=anchor_world,
        frame_rotation=frame_rotation,
        target_z=target_z,
        preferred_packing_xyzw=_live_held_tool_packing_xyzw(
            request, roll_rad=float(keyframe.roll_rad or 0.0)
        ),
    )
    if candidate is None:
        return keyframe
    try:
        raised = active_resolver.resolve(candidate)
    except GeometryResolutionError:
        return keyframe
    raised_z = float(raised.position_m[2])
    if raised_z < floor_z - 1e-4:
        return keyframe
    if ceiling_z is not None and raised_z > ceiling_z + 1e-4:
        return keyframe
    return candidate


def _hover_working_floor_z_m(request: MotionPlanRequest) -> float | None:
    """Minimum PRE_CONTACT / RETREAT height for held-tool sweep."""

    eef = request.world.robot_state.eef_pose
    holding = bool(
        request.world.robot_state.held_tool_id
        or request.world.robot_state.attached_object_id
    )
    if holding and eef is not None:
        eef_z = float(eef.position_m[2])
        if math.isfinite(eef_z):
            return eef_z
    support_z = _support_surface_z_m(request)
    if support_z is None:
        return None
    return support_z + _HOVER_MIN_ABOVE_SUPPORT_M


def _retarget_tcp_height(
    keyframe: RelativeKeyframeSpec,
    *,
    pose: Pose,
    anchor_world: np.ndarray,
    frame_rotation: np.ndarray,
    target_z: float,
    preferred_packing_xyzw: Sequence[float] | None = None,
) -> RelativeKeyframeSpec | None:
    z_m = float(pose.position_m[2])
    axis_local = np.asarray(keyframe.approach_axis_xyz, dtype=float)
    if (
        axis_local.shape == (3,)
        and np.all(np.isfinite(axis_local))
        and float(np.linalg.norm(axis_local)) >= 1e-9
    ):
        axis_local = axis_local / float(np.linalg.norm(axis_local))
        axis_world = frame_rotation @ axis_local
        axis_norm = float(np.linalg.norm(axis_world))
        if axis_norm >= 1e-9:
            axis_world = axis_world / axis_norm
            az = float(axis_world[2])
            if abs(az) >= 0.25:
                new_offset = (
                    float(keyframe.offset_along_approach_m)
                    + (target_z - z_m) / az
                )
                if math.isfinite(new_offset):
                    return keyframe.model_copy(
                        update={
                            "offset_along_approach_m": new_offset,
                            "metadata": {
                                **keyframe.metadata,
                                "held_tool_contact_height_canonicalized": True,
                                "held_tool_contact_height_from_offset_m": (
                                    keyframe.offset_along_approach_m
                                ),
                                "held_tool_contact_height_target_z_m": target_z,
                            },
                        }
                    )

    desired = np.asarray(
        (float(pose.position_m[0]), float(pose.position_m[1]), target_z),
        dtype=float,
    )
    delta = desired - anchor_world
    dist = float(np.linalg.norm(delta))
    if not math.isfinite(dist) or dist < 1e-6:
        return None
    axis_world = delta / dist
    axis_local = frame_rotation.T @ axis_world
    packing = keyframe.metadata.get("packing_orientation_xyzw")
    if packing is None and preferred_packing_xyzw is not None:
        packing = list(preferred_packing_xyzw)
    if packing is None:
        packing = list(pose.orientation_xyzw)
    return keyframe.model_copy(
        update={
            "approach_axis_xyz": tuple(float(v) for v in axis_local),
            "offset_along_approach_m": dist,
            "metadata": {
                **keyframe.metadata,
                "packing_orientation_xyzw": packing,
                "held_tool_contact_height_canonicalized": True,
                "held_tool_contact_height_retargeted_xy": True,
                "held_tool_contact_height_from_offset_m": (
                    keyframe.offset_along_approach_m
                ),
                "held_tool_contact_height_target_z_m": target_z,
            },
        }
    )


def canonicalize_sweep_strategy_heights(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    *,
    resolver: RelativePoseResolver | None = None,
) -> list[RelativeKeyframeSpec]:
    """Per-keyframe height fix, then lift PRE_CONTACT/RETREAT above CONTACT.

    Strategy validation requires hover phases at/above max CONTACT height. After
    individual CONTACT lifts, PRE_CONTACT can still sit below that plane; raise
    hover keyframes to the shared working floor before rejecting the strategy.
    """

    if not is_tool_act_contact_geometry_scope(request) or not keyframes:
        return list(keyframes)
    active_resolver = resolver or RelativePoseResolver(request.world)
    lifted = [
        canonicalize_contact_tcp_height(request, keyframe, resolver=active_resolver)
        for keyframe in keyframes
    ]
    contact_heights: list[float] = []
    for keyframe in lifted:
        if keyframe.keyframe_type not in _CONTACT_PHASE_TYPES:
            continue
        try:
            contact_heights.append(
                float(active_resolver.resolve(keyframe).position_m[2])
            )
        except GeometryResolutionError:
            continue
    hover_floor = _hover_working_floor_z_m(request)
    if contact_heights:
        max_contact = max(contact_heights)
        hover_floor = (
            max_contact if hover_floor is None else max(hover_floor, max_contact)
        )
    support_z = _support_surface_z_m(request)
    if support_z is not None:
        support_hover = support_z + _HOVER_MIN_ABOVE_SUPPORT_M
        hover_floor = (
            support_hover
            if hover_floor is None
            else max(hover_floor, support_hover)
        )
    if hover_floor is None:
        result = lifted
    else:
        result = []
        for keyframe in lifted:
            if keyframe.keyframe_type not in _HOVER_PHASE_TYPES:
                result.append(keyframe)
                continue
            try:
                pose = active_resolver.resolve(keyframe)
                anchor_world, frame_rotation = _frame_anchor_world(
                    active_resolver, keyframe
                )
            except GeometryResolutionError:
                result.append(keyframe)
                continue
            if float(pose.position_m[2]) >= hover_floor - 1e-6:
                result.append(keyframe)
                continue
            raised = _retarget_tcp_height(
                keyframe,
                pose=pose,
                anchor_world=anchor_world,
                frame_rotation=frame_rotation,
                target_z=hover_floor,
                preferred_packing_xyzw=_live_held_tool_packing_xyzw(
                    request, roll_rad=float(keyframe.roll_rad or 0.0)
                ),
            )
            result.append(keyframe if raised is None else raised)
    result = canonicalize_sweep_contact_start_over_targets(
        request, result, resolver=active_resolver
    )
    result = canonicalize_sweep_lateral_toward_region(
        request, result, resolver=active_resolver
    )
    return [
        canonicalize_held_tool_orientation_for_contact(
            request, keyframe, resolver=active_resolver
        )
        for keyframe in result
    ]


def _sweep_target_centroid_xy_m(request: MotionPlanRequest) -> np.ndarray | None:
    held_ids = _held_object_ids(request)
    points: list[np.ndarray] = []
    for target_id in request.task.target_ids:
        if not isinstance(target_id, str) or not target_id.strip():
            continue
        object_id = target_id.strip()
        if object_id in held_ids:
            continue
        record = request.world.objects.get(object_id)
        if not isinstance(record, Mapping):
            continue
        pose = record.get("pose")
        if not isinstance(pose, Mapping):
            continue
        position = _finite_vector(pose.get("position_m"), 3)
        if position is None:
            continue
        points.append(np.asarray(position[:2], dtype=float))
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def canonicalize_sweep_contact_start_over_targets(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    *,
    resolver: RelativePoseResolver | None = None,
) -> list[RelativeKeyframeSpec]:
    """Snap CONTACT_START onto a sweep-ready XY over the target cluster.

    Flat tools: start on the target centroid so the face covers the group.
    Hollow dishes: shift the start away from the goal region by the rim inner
    radius so the trailing rim begins at the cluster and can side-plow toward
    the region (AABB-centered starts leave blocks in the empty disk).
    """

    if not is_tool_act_contact_geometry_scope(request) or not keyframes:
        return list(keyframes)
    centroid = _sweep_target_centroid_xy_m(request)
    if centroid is None:
        return list(keyframes)
    desired_xy = np.asarray(centroid, dtype=float)
    metadata_flag = "sweep_contact_start_centroid"
    rim_r = _held_tool_hollow_rim_inner_radius_m(request)
    region_xy = _goal_region_center_xy_m(request)
    if rim_r is not None and region_xy is not None and float(rim_r) > 1e-4:
        away = desired_xy - np.asarray(region_xy, dtype=float)
        away_norm = float(np.linalg.norm(away))
        if away_norm > 1e-6:
            desired_xy = desired_xy + (away / away_norm) * float(rim_r)
            metadata_flag = "sweep_hollow_rim_plow_start"

    active_resolver = resolver or RelativePoseResolver(request.world)
    start_index = next(
        (
            index
            for index, keyframe in enumerate(keyframes)
            if keyframe.keyframe_type is KeyframeType.CONTACT_START
        ),
        None,
    )
    if start_index is None:
        return list(keyframes)
    try:
        start_pose = active_resolver.resolve(keyframes[start_index])
    except GeometryResolutionError:
        return list(keyframes)
    start_xy = np.asarray(start_pose.position_m[:2], dtype=float)
    if float(np.linalg.norm(start_xy - desired_xy)) < _CONTACT_START_CENTROID_SNAP_M:
        return list(keyframes)

    engagement = _contact_engagement_tcp_z_m(request)
    start_z = (
        float(engagement)
        if engagement is not None
        else float(start_pose.position_m[2])
    )
    packing = _live_held_tool_packing_xyzw(
        request, roll_rad=float(keyframes[start_index].roll_rad or 0.0)
    )
    if packing is None:
        packing = list(start_pose.orientation_xyzw)

    result = list(keyframes)
    retargeted = _retarget_tcp_world_xy(
        result[start_index],
        resolver=active_resolver,
        target_xy=desired_xy,
        target_z=start_z,
        packing_xyzw=packing,
        metadata_flag=metadata_flag,
    )
    if retargeted is None:
        return result
    result[start_index] = retargeted

    hover_z = _hover_working_floor_z_m(request)
    if hover_z is None:
        hover_z = start_z + _HOVER_MIN_ABOVE_SUPPORT_M
    hover_z = max(float(hover_z), start_z + _HOVER_MIN_ABOVE_SUPPORT_M)
    for index, keyframe in enumerate(result):
        if keyframe.keyframe_type is not KeyframeType.PRE_CONTACT:
            continue
        packing_pre = _live_held_tool_packing_xyzw(
            request, roll_rad=float(keyframe.roll_rad or 0.0)
        )
        if packing_pre is None:
            packing_pre = packing
        raised = _retarget_tcp_world_xy(
            keyframe,
            resolver=active_resolver,
            target_xy=desired_xy,
            target_z=hover_z,
            packing_xyzw=packing_pre,
            metadata_flag="sweep_pre_contact_follow_start",
        )
        if raised is not None:
            result[index] = raised
    return result


def canonicalize_sweep_lateral_toward_region(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    *,
    resolver: RelativePoseResolver | None = None,
) -> list[RelativeKeyframeSpec]:
    """Stretch a collapsed CONTACT path toward the goal region in XY.

    VLMs often emit CONTACT_START/SWEEP/END on the same target top (zero lateral
    travel). That plans and executes but cannot move footprints into
    ``target_region_id``. When a region exists, rewrite SWEEP/END (and RETREAT
    XY) along start→region while keeping the engagement height.
    """

    if not is_tool_act_contact_geometry_scope(request) or not keyframes:
        return list(keyframes)
    region_xy = _goal_region_center_xy_m(request)
    if region_xy is None:
        return list(keyframes)

    active_resolver = resolver or RelativePoseResolver(request.world)
    start_index = next(
        (
            index
            for index, keyframe in enumerate(keyframes)
            if keyframe.keyframe_type is KeyframeType.CONTACT_START
        ),
        None,
    )
    if start_index is None:
        return list(keyframes)
    try:
        start_pose = active_resolver.resolve(keyframes[start_index])
    except GeometryResolutionError:
        return list(keyframes)
    start_xy = np.asarray(start_pose.position_m[:2], dtype=float)
    start_z = float(start_pose.position_m[2])
    to_region = region_xy - start_xy
    dist_region = float(np.linalg.norm(to_region))
    if not math.isfinite(dist_region) or dist_region < _MIN_SWEEP_LATERAL_M:
        return list(keyframes)
    direction = to_region / dist_region

    end_index = next(
        (
            index
            for index, keyframe in enumerate(keyframes)
            if keyframe.keyframe_type is KeyframeType.CONTACT_END
        ),
        None,
    )
    if end_index is None:
        return list(keyframes)
    try:
        end_pose = active_resolver.resolve(keyframes[end_index])
    except GeometryResolutionError:
        return list(keyframes)
    end_xy = np.asarray(end_pose.position_m[:2], dtype=float)
    lateral = float(np.linalg.norm(end_xy - start_xy))
    progress = float(np.dot(end_xy - start_xy, direction))
    min_progress = max(
        _MIN_SWEEP_LATERAL_M,
        _MIN_SWEEP_REGION_PROGRESS_FRAC * dist_region,
    )
    if lateral >= _MIN_SWEEP_LATERAL_M and progress >= min_progress:
        return list(keyframes)

    move_indices = [
        index
        for index, keyframe in enumerate(keyframes)
        if keyframe.keyframe_type
        in {KeyframeType.CONTACT_SWEEP, KeyframeType.CONTACT_END}
    ]
    if not move_indices:
        return list(keyframes)

    packing = _live_held_tool_packing_xyzw(
        request, roll_rad=float(keyframes[start_index].roll_rad or 0.0)
    )
    if packing is None:
        packing = list(start_pose.orientation_xyzw)

    result = list(keyframes)
    for step, index in enumerate(move_indices, start=1):
        frac = float(step) / float(len(move_indices))
        target_xy = start_xy + direction * (dist_region * frac)
        retargeted = _retarget_tcp_world_xy(
            result[index],
            resolver=active_resolver,
            target_xy=target_xy,
            target_z=start_z,
            packing_xyzw=packing,
            metadata_flag="sweep_lateral_canonicalized",
        )
        if retargeted is not None:
            result[index] = retargeted

    # Keep RETREAT above the final contact XY so withdraw does not fly back
    # over the pre-sweep cluster.
    try:
        end_pose = active_resolver.resolve(result[end_index])
        end_xy = np.asarray(end_pose.position_m[:2], dtype=float)
        end_z = float(end_pose.position_m[2])
    except GeometryResolutionError:
        return result
    hover_z = _hover_working_floor_z_m(request)
    if hover_z is None:
        hover_z = end_z + _HOVER_MIN_ABOVE_SUPPORT_M
    hover_z = max(float(hover_z), end_z + _HOVER_MIN_ABOVE_SUPPORT_M)
    for index, keyframe in enumerate(result):
        if keyframe.keyframe_type is not KeyframeType.RETREAT:
            continue
        packing_retreat = _live_held_tool_packing_xyzw(
            request, roll_rad=float(keyframe.roll_rad or 0.0)
        )
        if packing_retreat is None:
            packing_retreat = packing
        retargeted = _retarget_tcp_world_xy(
            keyframe,
            resolver=active_resolver,
            target_xy=end_xy,
            target_z=hover_z,
            packing_xyzw=packing_retreat,
            metadata_flag="sweep_lateral_retreat_follow",
        )
        if retargeted is not None:
            result[index] = retargeted
    return result


def _retarget_tcp_world_xy(
    keyframe: RelativeKeyframeSpec,
    *,
    resolver: RelativePoseResolver,
    target_xy: np.ndarray,
    target_z: float,
    packing_xyzw: Sequence[float],
    metadata_flag: str,
) -> RelativeKeyframeSpec | None:
    try:
        anchor_world, frame_rotation = _frame_anchor_world(resolver, keyframe)
    except GeometryResolutionError:
        return None
    desired = np.asarray(
        (float(target_xy[0]), float(target_xy[1]), float(target_z)),
        dtype=float,
    )
    delta = desired - anchor_world
    dist = float(np.linalg.norm(delta))
    if not math.isfinite(dist) or dist < 1e-6:
        return None
    axis_world = delta / dist
    axis_local = frame_rotation.T @ axis_world
    return keyframe.model_copy(
        update={
            "approach_axis_xyz": tuple(float(v) for v in axis_local),
            "offset_along_approach_m": dist,
            "metadata": {
                **keyframe.metadata,
                "packing_orientation_xyzw": list(packing_xyzw),
                metadata_flag: True,
                "sweep_lateral_target_xy_m": [
                    float(target_xy[0]),
                    float(target_xy[1]),
                ],
            },
        }
    )


def _goal_region_center_xy_m(request: MotionPlanRequest) -> np.ndarray | None:
    region_id = request.task.goal.target_region_id
    if not region_id:
        return None
    record = request.world.objects.get(str(region_id))
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    if position is None:
        return None
    return np.asarray(position[:2], dtype=float)


def _reject_degenerate_sweep_lateral(
    request: MotionPlanRequest,
    keyframes: Sequence[RelativeKeyframeSpec],
    resolver: RelativePoseResolver,
) -> None:
    start_pose = None
    end_pose = None
    for keyframe in keyframes:
        if keyframe.keyframe_type is KeyframeType.CONTACT_START:
            start_pose = resolver.resolve(keyframe)
        elif keyframe.keyframe_type is KeyframeType.CONTACT_END:
            end_pose = resolver.resolve(keyframe)
    if start_pose is None or end_pose is None:
        return
    start_xy = np.asarray(start_pose.position_m[:2], dtype=float)
    end_xy = np.asarray(end_pose.position_m[:2], dtype=float)
    lateral = float(np.linalg.norm(end_xy - start_xy))
    if lateral < _MIN_SWEEP_LATERAL_M:
        raise ContactKeyframeGeometryError(
            f"sweep CONTACT_START→CONTACT_END lateral travel "
            f"{lateral:.3f}m is below {_MIN_SWEEP_LATERAL_M:.3f}m "
            "(collapsed contact path cannot move targets)"
        )
    region_xy = _goal_region_center_xy_m(request)
    if region_xy is None:
        return
    to_region = region_xy - start_xy
    dist_region = float(np.linalg.norm(to_region))
    if dist_region < _MIN_SWEEP_LATERAL_M:
        return
    direction = to_region / dist_region
    progress = float(np.dot(end_xy - start_xy, direction))
    min_progress = max(
        _MIN_SWEEP_LATERAL_M,
        _MIN_SWEEP_REGION_PROGRESS_FRAC * dist_region,
    )
    if progress < min_progress:
        raise ContactKeyframeGeometryError(
            f"sweep CONTACT_END progress toward region "
            f"{progress:.3f}m is below {min_progress:.3f}m "
            f"(region={request.task.goal.target_region_id!r})"
        )


def _is_trivial_two_point_transfer(
    kinds: Sequence[KeyframeType],
    keyframes: Sequence[RelativeKeyframeSpec],
) -> bool:
    if len(keyframes) != 2:
        return False
    if not all(kind is KeyframeType.TRANSFER for kind in kinds):
        return False
    return True


def _require_ordered_sweep_phases(kinds: Sequence[KeyframeType]) -> None:
    expected = (
        KeyframeType.PRE_CONTACT,
        KeyframeType.CONTACT_START,
        KeyframeType.CONTACT_SWEEP,
        KeyframeType.CONTACT_END,
        KeyframeType.RETREAT,
    )
    index = 0
    phase_index = 0
    while index < len(kinds) and phase_index < len(expected):
        current = kinds[index]
        want = expected[phase_index]
        if current is want:
            if want is KeyframeType.CONTACT_SWEEP:
                while index < len(kinds) and kinds[index] is KeyframeType.CONTACT_SWEEP:
                    index += 1
            else:
                index += 1
            phase_index += 1
            continue
        raise ContactKeyframeGeometryError(
            "sweep strategy must follow PRE_CONTACT → CONTACT_START → "
            "CONTACT_SWEEP+ → CONTACT_END → RETREAT in order "
            f"(saw {' → '.join(kind.value for kind in kinds)})"
        )
    if phase_index < len(expected) or index != len(kinds):
        raise ContactKeyframeGeometryError(
            "sweep strategy must follow PRE_CONTACT → CONTACT_START → "
            "CONTACT_SWEEP+ → CONTACT_END → RETREAT in order "
            f"(saw {' → '.join(kind.value for kind in kinds)})"
        )


def _held_tool_half_height_m(request: MotionPlanRequest) -> float:
    for object_id in sorted(_held_object_ids(request)):
        record = request.world.objects.get(object_id)
        if not isinstance(record, Mapping):
            continue
        dims = _finite_vector(record.get("dimensions_m"), 3)
        if dims is not None:
            return 0.5 * float(dims[2])
    return 0.0


def _reject_inverted_held_tool_orientation(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    pose: Pose,
) -> None:
    """Reject contact poses that flip the live held-tool TCP axis.

    After acquire, vacuum/gripper sweeps keep roughly the same tool-axis
    attitude. VLM proposals that invert it (e.g. ``tool_axis_to_align=+z``
    while the live EEF points ``-z`` down) routinely exhaust IK without being
    spatially near the world origin.
    """

    if not (
        request.world.robot_state.held_tool_id
        or request.world.robot_state.attached_object_id
    ):
        return
    eef = request.world.robot_state.eef_pose
    if eef is None:
        return
    live_z = _pose_tool_z_world(eef.orientation_xyzw)
    target_z = _pose_tool_z_world(pose.orientation_xyzw)
    if live_z is None or target_z is None:
        return
    dot = float(np.dot(live_z, target_z))
    if dot < _MIN_HELD_TOOL_AXIS_DOT:
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: resolved TCP tool-z is inverted vs the "
            f"live held-tool EEF (dot={dot:.3f}, tool_axis_to_align="
            f"{keyframe.tool_axis_to_align!r}); keep the grasp approach axis "
            "for contact tool_act"
        )


def _pose_tool_z_world(
    orientation_xyzw: Sequence[float] | None,
) -> np.ndarray | None:
    if orientation_xyzw is None:
        return None
    values = _finite_vector(orientation_xyzw, 4)
    if values is None:
        return None
    rotation = _quaternion_matrix_xyzw(values)
    axis = rotation[:, 2]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm < 1e-12:
        return None
    return axis / norm


def _live_held_tool_packing_xyzw(
    request: MotionPlanRequest,
    *,
    roll_rad: float = 0.0,
) -> list[float] | None:
    """Live EEF quaternion, optionally rolled about tool +z."""

    if not (
        request.world.robot_state.held_tool_id
        or request.world.robot_state.attached_object_id
    ):
        return None
    eef = request.world.robot_state.eef_pose
    if eef is None:
        return None
    live_q = _finite_vector(eef.orientation_xyzw, 4)
    if live_q is None:
        return None
    rotation = _quaternion_matrix_xyzw(live_q)
    roll = float(roll_rad)
    if abs(roll) > 1e-9 and math.isfinite(roll):
        cos_r = math.cos(roll)
        sin_r = math.sin(roll)
        roll_m = np.asarray(
            (
                (cos_r, -sin_r, 0.0),
                (sin_r, cos_r, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        rotation = rotation @ roll_m
    return list(_matrix_quaternion_xyzw(rotation))


def _reject_outside_workspace(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    position: np.ndarray,
) -> None:
    eef = request.world.robot_state.eef_pose
    if eef is None:
        return
    eef_position = np.asarray(eef.position_m, dtype=float)
    if eef_position.shape != (3,) or not np.all(np.isfinite(eef_position)):
        return
    distance = float(np.linalg.norm(position - eef_position))
    if distance > _MAX_EEF_DISTANCE_M:
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: resolved TCP is "
            f"{distance:.3f}m from current EEF (limit={_MAX_EEF_DISTANCE_M}m)"
        )


def _reject_task_unrelated(
    request: MotionPlanRequest,
    keyframe: RelativeKeyframeSpec,
    position: np.ndarray,
) -> None:
    landmarks = _task_landmark_positions(request)
    if not landmarks:
        return
    nearest = min(
        float(np.linalg.norm(position - landmark)) for landmark in landmarks
    )
    if nearest > _TASK_RELEVANCE_RADIUS_M:
        raise ContactKeyframeGeometryError(
            f"{keyframe.keyframe_id}: resolved TCP is {nearest:.3f}m from "
            f"nearest tool/target/region landmark "
            f"(limit={_TASK_RELEVANCE_RADIUS_M}m); "
            f"position_m={_fmt(position)}"
        )


def _task_landmark_ids(request: MotionPlanRequest) -> list[str]:
    ids: list[str] = []
    tool = request.task.tool
    if isinstance(tool, str) and tool.strip():
        ids.append(tool.strip())
    for target_id in request.task.target_ids:
        if isinstance(target_id, str) and target_id.strip():
            ids.append(target_id.strip())
    goal = request.task.goal
    for candidate in (goal.target_object_id, goal.target_region_id):
        if isinstance(candidate, str) and candidate.strip():
            ids.append(candidate.strip())
    # Preserve order while dropping duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for object_id in ids:
        if object_id not in seen:
            seen.add(object_id)
            unique.append(object_id)
    return unique


def _task_landmark_positions(request: MotionPlanRequest) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for object_id in _task_landmark_ids(request):
        center = _object_center_world(request.world.objects.get(object_id))
        if center is not None:
            positions.append(center)
    return positions


def _support_surface_z_m(request: MotionPlanRequest) -> float | None:
    """Estimate the resting support/table height for contact motions.

    The held tool is intentionally excluded: after acquire it sits above the
    table, and using its bottom as the floor rejects valid table-level sweep
    poses (TCP near target height reads as "below support").
    """

    held_ids = _held_object_ids(request)
    table_candidates: list[float] = []
    resting_candidates: list[float] = []

    for obstacle in request.world.obstacles:
        if not isinstance(obstacle, Mapping):
            continue
        label = f"{obstacle.get('id', '')} {obstacle.get('kind', '')}".lower()
        if not any(token in label for token in ("table", "counter", "support")):
            continue
        maximum = _finite_vector(obstacle.get("aabb_max_m"), 3)
        if maximum is not None:
            table_candidates.append(float(maximum[2]))

    for object_id in _task_landmark_ids(request):
        if object_id in held_ids:
            continue
        bottom = _object_bottom_z(request.world.objects.get(object_id))
        if bottom is not None:
            resting_candidates.append(bottom)

    if table_candidates:
        # Prefer an explicit table/counter top when present.
        return max(table_candidates)
    if resting_candidates:
        # Targets / collection zones rest on the support; use their bottoms.
        return max(resting_candidates)
    return None


def _held_object_ids(request: MotionPlanRequest) -> set[str]:
    """Object ids that should not define the table support height."""

    held: set[str] = set()
    tool = request.task.tool
    if isinstance(tool, str) and tool.strip():
        held.add(tool.strip())
    state = request.world.robot_state
    for candidate in (state.held_tool_id, state.attached_object_id):
        if isinstance(candidate, str) and candidate.strip():
            held.add(candidate.strip())
    return held


def _object_center_world(record: Any) -> np.ndarray | None:
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None:
        return None
    if orientation is None:
        return position
    rotation = _quaternion_matrix_xyzw(orientation)
    anchors = record.get("anchors", {})
    if isinstance(anchors, Mapping):
        center_local = _finite_vector(anchors.get("center"), 3)
        if center_local is not None:
            return position + rotation @ center_local
    return position


def _object_bottom_z(record: Any) -> float | None:
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    rotation = _quaternion_matrix_xyzw(orientation)
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    bottom_local = _finite_vector(anchors.get("bottom"), 3)
    dims = _finite_vector(record.get("dimensions_m"), 3)
    if bottom_local is None and dims is not None:
        bottom_local = np.array([0.0, 0.0, -0.5 * float(dims[2])], dtype=float)
    if bottom_local is None:
        return float(position[2])
    return float((position + rotation @ bottom_local)[2])


def _object_top_z(record: Any) -> float | None:
    if not isinstance(record, Mapping):
        return None
    pose = record.get("pose")
    if not isinstance(pose, Mapping):
        return None
    position = _finite_vector(pose.get("position_m"), 3)
    orientation = _finite_vector(pose.get("orientation_xyzw"), 4)
    if position is None or orientation is None:
        return None
    rotation = _quaternion_matrix_xyzw(orientation)
    anchors = record.get("anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    top_local = _finite_vector(anchors.get("top"), 3)
    dims = _finite_vector(record.get("dimensions_m"), 3)
    if top_local is None and dims is not None:
        top_local = np.array([0.0, 0.0, 0.5 * float(dims[2])], dtype=float)
    if top_local is None:
        return float(position[2])
    return float((position + rotation @ top_local)[2])


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _quaternion_matrix_xyzw(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in values)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _fmt(position: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in position]


__all__ = [
    "ContactKeyframeGeometryError",
    "canonicalize_contact_tcp_height",
    "canonicalize_held_tool_axis_for_contact",
    "canonicalize_held_tool_orientation_for_contact",
    "canonicalize_sweep_contact_start_over_targets",
    "canonicalize_sweep_lateral_toward_region",
    "canonicalize_sweep_strategy_heights",
    "contact_keyframe_geometry_errors",
    "is_tool_act_contact_geometry_scope",
    "validate_resolved_contact_keyframe",
    "validate_sweep_keyframe_strategy",
]
