"""Contact tool_act keyframe geometry is rejected before IK, without task hardcoding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
from tuj.m5_motion.contact_keyframe_validation import (
    ContactKeyframeGeometryError,
    is_tool_act_contact_geometry_scope,
    validate_resolved_contact_keyframe,
    validate_sweep_keyframe_strategy,
)
from tuj.m5_motion.geometry import RelativePoseResolver
from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    ContactManipulationSpec,
    GoalType,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionConstraints,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    Pose,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    WorldSnapshot,
)
from tuj.m5_motion.strategy import EdgePlanResult
from tuj.m5_motion.vlm_provider import (
    GeneratedKeyframe,
    GeneratedKeyframeBatch,
    GeneratedStrategy,
    NoValidKeyframeCandidatesError,
    OpenAIKeyframeProvider,
    OpenAIKeyframeProviderConfig,
    _phase_contract_payload,
)


def _object(
    position: tuple[float, float, float],
    *,
    dimensions: tuple[float, float, float] = (0.08, 0.08, 0.04),
) -> dict:
    return {
        "pose": {
            "position_m": list(position),
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "dimensions_m": list(dimensions),
        "anchors": {"center": [0.0, 0.0, 0.0], "origin": [0.0, 0.0, 0.0]},
    }


def _sweep_request(
    *,
    action_type: str = "tool_act",
    primitive: str = "sweep",
    with_contact: bool = True,
) -> MotionPlanRequest:
    contact = None
    if with_contact:
        contact = ContactManipulationSpec(
            primitive=primitive,
            contact_surface="AUTO",
        )
    return MotionPlanRequest(
        request_id="request-sweep",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="task-planner-1",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene-sweep"),
            robot_state=RobotState(
                robot_id="ur5e",
                joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
                joint_positions_rad=[0.0, -1.2, 1.4, -1.5, -1.5, 0.0],
                eef_pose=Pose(
                    frame_id="world",
                    position_m=(-0.24, -0.10, 0.99),
                    # Live vacuum attitude after acquire: TCP +z points down.
                    orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
            objects={
                # Held tool underside near the live EEF (thin tool below TCP).
                "tool_pusher": _object(
                    (-0.20, -0.05, 0.985), dimensions=(0.12, 0.12, 0.01)
                ),
                "block_a": _object((-0.10, 0.05, 0.84)),
                "collect_zone": _object(
                    (0.15, 0.10, 0.82), dimensions=(0.20, 0.20, 0.02)
                ),
            },
            obstacles=[
                {
                    "id": "table",
                    "kind": "box",
                    "aabb_min_m": [-0.6, -0.5, 0.0],
                    "aabb_max_m": [0.6, 0.5, 0.80],
                }
            ],
        ),
        task=MotionTask(
            task_id="sweep-blocks",
            subgoal_id="sg-sweep",
            action_type=action_type,
            ee="vac",
            tool="tool_pusher",
            target_ids=["block_a"],
            contact=contact,
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_region_id="collect_zone",
                target_pose=Pose(
                    frame_id="world",
                    position_m=(0.15, 0.10, 0.86),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ),
        constraints=MotionConstraints(),
    )


def _keyframe(
    *,
    keyframe_id: str,
    frame_ref: str,
    anchor: str = "center",
    offset: float = 0.05,
    approach: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tool_axis: str = "-z",
    keyframe_type: KeyframeType = KeyframeType.TRANSFER,
) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=keyframe_id,
        keyframe_type=keyframe_type,
        frame_ref=frame_ref,
        anchor=anchor,
        approach_axis_xyz=approach,
        tool_axis_to_align=tool_axis,
        offset_along_approach_m=offset,
        roll_rad=0.0,
        planner=KeyframePlannerType.CARTESIAN,
    )


def _valid_sweep_sequence() -> list[RelativeKeyframeSpec]:
    """PRE_CONTACT at EEF → CONTACT_* at engagement → RETREAT hover.

    Fixture: block top z=0.86, tool-below-TCP≈0.01 → engagement TCP≈0.87;
    live EEF z=0.99 for PRE/RETREAT.
    """

    return [
        _keyframe(
            keyframe_id="pre",
            frame_ref="object:block_a",
            offset=0.15,
            keyframe_type=KeyframeType.PRE_CONTACT,
        ),
        _keyframe(
            keyframe_id="start",
            frame_ref="object:block_a",
            offset=0.03,
            keyframe_type=KeyframeType.CONTACT_START,
        ),
        _keyframe(
            keyframe_id="sweep",
            frame_ref="object:block_a",
            offset=0.03,
            keyframe_type=KeyframeType.CONTACT_SWEEP,
        ),
        _keyframe(
            keyframe_id="end",
            frame_ref="object:collect_zone",
            offset=0.05,
            keyframe_type=KeyframeType.CONTACT_END,
        ),
        _keyframe(
            keyframe_id="retract",
            frame_ref="object:collect_zone",
            offset=0.17,
            keyframe_type=KeyframeType.RETREAT,
        ),
    ]


def _provenance() -> StrategyGenerationProvenance:
    return StrategyGenerationProvenance(
        generator_kind=StrategyGeneratorKind.TEMPLATE,
        generator_id="test",
        input_hash="hash",
    )


def test_scope_is_tool_act_sweep_only() -> None:
    assert is_tool_act_contact_geometry_scope(_sweep_request())
    assert not is_tool_act_contact_geometry_scope(
        _sweep_request(action_type="PICK", with_contact=False)
    )
    assert not is_tool_act_contact_geometry_scope(
        _sweep_request(primitive="push")
    )
    transport = _sweep_request(action_type="TRANSPORT", with_contact=False)
    assert not is_tool_act_contact_geometry_scope(transport)


def test_rejects_world_origin_transfer() -> None:
    request = _sweep_request()
    keyframe = _keyframe(
        keyframe_id="t1",
        frame_ref="world",
        anchor="origin",
        offset=0.05,
        approach=(0.0, -1.0, 0.0),
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    with pytest.raises(ContactKeyframeGeometryError, match="world origin"):
        validate_resolved_contact_keyframe(request, keyframe, pose)


def test_rejects_pose_below_support_surface() -> None:
    request = _sweep_request()
    # Anchor on the block but push deeply through the table.
    keyframe = _keyframe(
        keyframe_id="t_below",
        frame_ref="object:block_a",
        anchor="center",
        offset=1.0,
        approach=(0.0, 0.0, -1.0),
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    with pytest.raises(ContactKeyframeGeometryError, match="below support"):
        validate_resolved_contact_keyframe(request, keyframe, pose)


def test_allows_near_table_sweep_within_support_margin() -> None:
    request = _sweep_request()
    # Table top is 0.80; a pose ~6cm below block center still counts as
    # near-contact, not a world-floor / through-floor failure.
    keyframe = _keyframe(
        keyframe_id="t_near_table",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.10,
        approach=(0.0, 0.0, -1.0),
        tool_axis="+z",
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    assert 0.70 < float(pose.position_m[2]) < 0.80
    validate_resolved_contact_keyframe(request, keyframe, pose)


def test_rejects_inverted_held_tool_orientation() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Live vacuum attitude: TCP +z points world -z (quat x=1).
    request.world.robot_state.eef_pose = Pose(
        frame_id="world",
        position_m=(-0.24, -0.10, 0.99),
        orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
    )
    # VLM +z alignment with world-up approach → identity (inverted vs live).
    keyframe = _keyframe(
        keyframe_id="t_flip",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.04,
        approach=(0.0, 0.0, 1.0),
        tool_axis="+z",
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    with pytest.raises(ContactKeyframeGeometryError, match="inverted"):
        validate_resolved_contact_keyframe(request, keyframe, pose)


def test_canonicalizes_inverted_held_tool_axis_to_matching() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_held_tool_axis_for_contact,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    request.world.robot_state.eef_pose = Pose(
        frame_id="world",
        position_m=(-0.24, -0.10, 0.99),
        orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
    )
    keyframe = _keyframe(
        keyframe_id="t_flip",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.04,
        approach=(0.0, 0.0, 1.0),
        tool_axis="+z",
    )
    fixed = canonicalize_held_tool_axis_for_contact(request, keyframe)
    assert fixed.tool_axis_to_align == "-z"
    assert fixed.metadata.get("held_tool_axis_canonicalized") is True
    pose = RelativePoseResolver(request.world).resolve(fixed)
    validate_resolved_contact_keyframe(request, fixed, pose)


def test_accepts_matching_held_tool_orientation() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    request.world.robot_state.eef_pose = Pose(
        frame_id="world",
        position_m=(-0.24, -0.10, 0.99),
        orientation_xyzw=(1.0, 0.0, 0.0, 0.0),
    )
    keyframe = _keyframe(
        keyframe_id="t_match",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.04,
        approach=(0.0, 0.0, 1.0),
    )
    keyframe = keyframe.model_copy(update={"tool_axis_to_align": "-z"})
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    validate_resolved_contact_keyframe(request, keyframe, pose)


def test_rejects_pose_far_from_task_landmarks() -> None:
    request = _sweep_request()
    request.world.robot_state.eef_pose = None
    # World-frame pose above the table but far from tool/targets/region.
    keyframe = _keyframe(
        keyframe_id="t_far",
        frame_ref="world",
        anchor="origin",
        offset=2.0,
        approach=(0.70710678118, 0.0, 0.70710678118),
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    with pytest.raises(ContactKeyframeGeometryError, match="landmark"):
        validate_resolved_contact_keyframe(request, keyframe, pose)


def test_accepts_transfer_near_sweep_target() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    keyframe = _keyframe(
        keyframe_id="t_ok",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.04,
        approach=(0.0, 0.0, 1.0),
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    validate_resolved_contact_keyframe(request, keyframe, pose)


def test_held_tool_height_does_not_raise_support_floor() -> None:
    """After acquire the tool floats above the table; that is not the support."""

    request = _sweep_request()
    # Held tool pose after lift — bottom near 0.98m if included in support.
    request.world.objects["tool_pusher"] = _object(
        (-0.24, -0.10, 0.99), dimensions=(0.12, 0.12, 0.02)
    )
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    keyframe = _keyframe(
        keyframe_id="t_table",
        frame_ref="object:block_a",
        anchor="center",
        offset=0.02,
        approach=(0.0, 0.0, 1.0),
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    assert float(pose.position_m[2]) < 0.95
    validate_resolved_contact_keyframe(request, keyframe, pose)


def test_pick_request_ignores_world_origin_gate() -> None:
    request = _sweep_request(action_type="PICK", with_contact=False)
    request.task.target_ids = ["block_a"]
    request.task.goal = MotionGoal(
        goal_type=GoalType.POSE,
        target_object_id="block_a",
        target_pose=Pose(
            frame_id="world",
            position_m=(-0.10, 0.05, 0.84),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    keyframe = _keyframe(
        keyframe_id="ignored",
        frame_ref="world",
        anchor="origin",
        offset=0.0,
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    validate_resolved_contact_keyframe(request, keyframe, pose)


def test_rejects_trivial_two_point_transfer_strategy() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    keyframes = [
        _keyframe(keyframe_id="a", frame_ref="object:block_a", offset=0.03),
        _keyframe(keyframe_id="b", frame_ref="object:collect_zone", offset=0.03),
    ]
    with pytest.raises(ContactKeyframeGeometryError, match="trivial two-point"):
        validate_sweep_keyframe_strategy(request, keyframes)


def test_accepts_contact_sweep_sequence() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    validate_sweep_keyframe_strategy(request, _valid_sweep_sequence())


def test_rejects_contact_phase_too_high_above_support() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Far above engagement (~0.87): offset 0.40 → z≈1.24 outside band.
    keyframe = _keyframe(
        keyframe_id="high",
        frame_ref="object:block_a",
        offset=0.40,
        keyframe_type=KeyframeType.CONTACT_SWEEP,
    )
    pose = RelativePoseResolver(request.world).resolve(keyframe)
    with pytest.raises(ContactKeyframeGeometryError, match="contact band"):
        validate_resolved_contact_keyframe(request, keyframe, pose)


def test_canonicalizes_low_contact_height_into_engagement_band() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_contact_tcp_height,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Drop below engagement via negative standoff on +z approach (keeps tool axis).
    low = _keyframe(
        keyframe_id="too_low",
        frame_ref="object:block_a",
        offset=-0.08,
        approach=(0.0, 0.0, 1.0),
        keyframe_type=KeyframeType.CONTACT_SWEEP,
    )
    before = RelativePoseResolver(request.world).resolve(low)
    assert float(before.position_m[2]) < 0.80
    fixed = canonicalize_contact_tcp_height(request, low)
    assert fixed.metadata.get("held_tool_contact_height_canonicalized") is True
    pose = RelativePoseResolver(request.world).resolve(fixed)
    validate_resolved_contact_keyframe(request, fixed, pose)


def test_canonicalizes_high_contact_height_down_to_engagement() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_contact_tcp_height,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Post-grasp EEF height (~0.99) is above the engagement band.
    high = _keyframe(
        keyframe_id="lift_height",
        frame_ref="object:block_a",
        offset=0.15,
        keyframe_type=KeyframeType.CONTACT_SWEEP,
    )
    before = RelativePoseResolver(request.world).resolve(high)
    assert float(before.position_m[2]) > 0.95
    fixed = canonicalize_contact_tcp_height(request, high)
    assert fixed.metadata.get("held_tool_contact_height_canonicalized") is True
    after = RelativePoseResolver(request.world).resolve(fixed)
    assert float(after.position_m[2]) < 0.92
    validate_resolved_contact_keyframe(request, fixed, after)


def test_canonicalizes_horizontal_approach_low_contact_height() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_contact_tcp_height,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Drop the target so a horizontal center pose sits below engagement floor.
    request.world.objects["block_a"] = _object(
        (-0.10, 0.05, 0.805), dimensions=(0.08, 0.08, 0.02)
    )
    keyframe = _keyframe(
        keyframe_id="side_low",
        frame_ref="object:block_a",
        offset=0.0,
        approach=(1.0, 0.0, 0.0),
        tool_axis="+z",
        keyframe_type=KeyframeType.CONTACT_SWEEP,
    )
    before = RelativePoseResolver(request.world).resolve(keyframe)
    assert float(before.position_m[2]) < 0.82
    fixed = canonicalize_contact_tcp_height(request, keyframe)
    assert fixed.metadata.get("held_tool_contact_height_canonicalized") is True
    after = RelativePoseResolver(request.world).resolve(fixed)
    assert abs(float(after.position_m[0]) - float(before.position_m[0])) < 1e-6
    assert abs(float(after.position_m[1]) - float(before.position_m[1])) < 1e-6
    validate_resolved_contact_keyframe(request, fixed, after)


def test_canonicalizes_downward_approach_with_negative_offset() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_contact_tcp_height,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    # Approach -z with large offset: TCP below engagement; raising needs a
    # less-positive / negative offset along -z.
    keyframe = _keyframe(
        keyframe_id="down_low",
        frame_ref="object:block_a",
        offset=0.08,
        approach=(0.0, 0.0, -1.0),
        tool_axis="+z",
        keyframe_type=KeyframeType.CONTACT_SWEEP,
    )
    before = RelativePoseResolver(request.world).resolve(keyframe)
    assert float(before.position_m[2]) < 0.80
    fixed = canonicalize_contact_tcp_height(request, keyframe)
    assert fixed.metadata.get("held_tool_contact_height_canonicalized") is True
    assert float(fixed.offset_along_approach_m) < float(keyframe.offset_along_approach_m)
    after = RelativePoseResolver(request.world).resolve(fixed)
    validate_resolved_contact_keyframe(request, fixed, after)


def test_canonicalizes_low_pre_contact_to_eef_height() -> None:
    from tuj.m5_motion.contact_keyframe_validation import (
        canonicalize_contact_tcp_height,
        canonicalize_sweep_strategy_heights,
    )

    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    low_pre = _keyframe(
        keyframe_id="pre_low",
        frame_ref="object:block_a",
        offset=0.02,
        keyframe_type=KeyframeType.PRE_CONTACT,
    )
    fixed = canonicalize_contact_tcp_height(request, low_pre)
    assert fixed.metadata.get("held_tool_contact_height_canonicalized") is True
    pre_z = float(RelativePoseResolver(request.world).resolve(fixed).position_m[2])
    eef_z = float(request.world.robot_state.eef_pose.position_m[2])
    assert pre_z >= eef_z - 1e-4

    # Strategy-level: PRE below max CONTACT is lifted before validation.
    sequence = _valid_sweep_sequence()
    sequence[0] = low_pre
    raised = canonicalize_sweep_strategy_heights(request, sequence)
    validate_sweep_keyframe_strategy(request, raised)


def test_compiler_marks_invalid_sweep_without_ik() -> None:
    request = _sweep_request()
    request.world.robot_state.held_tool_id = "tool_pusher"
    request.world.robot_state.attached_object_id = "tool_pusher"
    ik_calls: list[tuple[float, float, float]] = []

    class _RecordingKinematics:
        def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
            ik_calls.append(tuple(float(value) for value in world_pos))
            return IKSolutionSet(
                solutions=(
                    IKResult(
                        solved=True,
                        qpos=(0.0, -1.0, 1.0, -1.0, -1.5, 0.0),
                        branch_id="ok",
                        position_error_m=0.0,
                        orientation_error_rad=0.0,
                    ),
                ),
                best_position_error_m=0.0,
                best_orientation_error_rad=0.0,
                attempted_seeds=1,
                enumeration_complete=True,
            )

    class _AlwaysValid:
        def __call__(self, qpos, keyframe):
            return True

    class _AlwaysEdge:
        def plan(self, source, target, source_keyframe, target_keyframe):
            return EdgePlanResult(
                valid=True,
                joint_path=(tuple(source), tuple(target)),
            )

    bad = KeyframePlanCandidate(
        strategy_id="bad-two-point",
        keyframes=[
            _keyframe(
                keyframe_id="t1",
                frame_ref="object:block_a",
                offset=0.03,
            ),
            _keyframe(
                keyframe_id="t1b",
                frame_ref="object:collect_zone",
                offset=0.03,
            ),
        ],
        rationale="trivial two-point transfer",
        provenance=_provenance(),
    )
    good = KeyframePlanCandidate(
        strategy_id="good-sweep",
        keyframes=_valid_sweep_sequence(),
        rationale="contact sweep sequence",
        provenance=_provenance(),
    )
    result = FirstFeasibleStrategyCompiler(_RecordingKinematics()).compile(
        request.world,
        [bad, good],
        start_joint_config=request.world.robot_state.joint_positions_rad,
        state_validator=_AlwaysValid(),
        edge_planner=_AlwaysEdge(),
        request=request,
    )
    assert result.attempts[0].failure_code == "KEYFRAME_GEOMETRY_INVALID"
    assert "trivial two-point" in result.attempts[0].detail
    assert result.solved
    assert result.connected is not None
    assert result.connected.strategy_id == "good-sweep"
    assert ik_calls, "valid strategy should still reach IK"
    assert all(abs(position[2]) > 0.5 for position in ik_calls)


def test_vlm_provider_rejects_world_origin_sweep_batch() -> None:
    request = _sweep_request()
    bad_strategy = GeneratedStrategy(
        strategy_id="bad",
        rationale="world origin transfer",
        keyframes=[
            GeneratedKeyframe(
                keyframe_id="t1",
                keyframe_type="TRANSFER",
                frame_ref="world",
                anchor="origin",
                approach_axis_xyz=[0.0, -1.0, 0.0],
                tool_axis_to_align="-z",
                offset_along_approach_m=0.05,
                roll_rad=0.0,
                planner="CARTESIAN",
            ),
            GeneratedKeyframe(
                keyframe_id="t2",
                keyframe_type="TRANSFER",
                frame_ref="world",
                anchor="origin",
                approach_axis_xyz=[0.0, -1.0, 0.0],
                tool_axis_to_align="-z",
                offset_along_approach_m=0.08,
                roll_rad=0.0,
                planner="CARTESIAN",
            ),
        ],
    )
    batch = GeneratedKeyframeBatch(
        candidates=[
            bad_strategy,
            bad_strategy.model_copy(
                update={
                    "strategy_id": "bad-alt",
                    "rationale": "alternate world origin transfer",
                }
            ),
        ]
    )

    class _FailingResponses:
        def __init__(self) -> None:
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                id=f"resp_{len(self.calls)}",
                status="completed",
                output_parsed=batch,
            )

    class _Client:
        def __init__(self) -> None:
            self.responses = _FailingResponses()

    client = _Client()
    provider = OpenAIKeyframeProvider(
        OpenAIKeyframeProviderConfig(
            model="gpt-test",
            candidate_count=2,
            cache_dir=None,
        ),
        client=client,
    )
    with pytest.raises(NoValidKeyframeCandidatesError):
        provider.generate(request)
    assert len(client.responses.calls) >= 2


def test_phase_contract_mentions_sweep_invalid_world_origin() -> None:
    payload = _phase_contract_payload(_sweep_request())
    assert payload["operation"] == "TOOL_ACT"
    assert payload["primitive"] == "sweep"
    assert payload["canonical_sequence"][0] == "PRE_CONTACT"
    assert any("two-point" in item.lower() for item in payload["invalid"])
    assert any("world" in item.lower() for item in payload["invalid"])


def test_contact_sweep_keyframe_types_map_to_segments() -> None:
    from tuj.m5_motion.plan_builder import _segment_type
    from tuj.m5_motion.schema import SegmentType

    assert _segment_type(KeyframeType.PRE_CONTACT) is SegmentType.APPROACH
    assert _segment_type(KeyframeType.CONTACT_START) is SegmentType.TRANSFER
    assert _segment_type(KeyframeType.CONTACT_SWEEP) is SegmentType.TRANSFER
    assert _segment_type(KeyframeType.CONTACT_END) is SegmentType.TRANSFER
    assert _segment_type(KeyframeType.RETREAT) is SegmentType.RETREAT
