"""Multi-finger tabletop enclosure rewrites support-penetrating approaches."""

from __future__ import annotations

import pytest

from tuj.m5_motion.grasp_geometry import (
    MULTI_FINGER_TABLETOP_ENCLOSURE,
    bind_multi_finger_tabletop_enclosure,
    bind_suction_surface_contact,
    multi_finger_tabletop_pre_grasp_standoff_candidates,
)
from tuj.m5_motion.schema import (
    ArtifactProvenance,
    GoalType,
    KeyframePlanArtifact,
    KeyframePlanCandidate,
    KeyframePlannerType,
    KeyframeType,
    ModuleName,
    MotionGoal,
    MotionPlanRequest,
    MotionTask,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    StrategyGenerationProvenance,
    StrategyGeneratorKind,
    WorldSnapshot,
)


def _pose(*, z: float = 0.96) -> dict:
    return {
        "frame_id": "world",
        "position_m": [2.5, -2.9, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _request(
    *,
    ee: str = "3F",
    capabilities: list[str] | None = None,
    object_z: float = 0.959,
    dims: tuple[float, float, float] = (0.08, 0.08, 0.08),
) -> MotionPlanRequest:
    half_z = dims[2] * 0.5
    caps = (
        capabilities
        if capabilities is not None
        else (["multi_finger_contact"] if ee == "3F" else ["suction"])
    )
    return MotionPlanRequest(
        request_id="request:mf-tabletop",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["j1"],
                joint_positions_rad=[0.0],
            ),
            objects={
                "fruit": {
                    "pose": _pose(z=object_z),
                    "dimensions_m": list(dims),
                    "anchors": {
                        "center": [0.0, 0.0, 0.0],
                        "top": [0.0, 0.0, half_z],
                        "top_center": [0.0, 0.0, half_z],
                    },
                }
            },
            obstacles=[
                {
                    "obstacle_id": "island_top",
                    "aabb_min_m": [2.0, -3.5, 0.89],
                    "aabb_max_m": [3.7, -2.5, 0.92],
                    "collision_enabled_in_source": True,
                }
            ],
        ),
        task=MotionTask(
            task_id="acquire-fruit",
            subgoal_id="sg",
            action_type="acquire",
            ee=ee,
            target_ids=["fruit"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="fruit",
            ),
            allowed_touch_objects=["fruit"],
            metadata={
                "operation": "ACQUIRE",
                "ee_capabilities": caps,
                "attach_target": True,
            },
        ),
    )


def _keyframe(
    *,
    kind: KeyframeType,
    approach: tuple[float, float, float],
    tool: str,
    offset: float,
    anchor: str = "center",
    kid: str | None = None,
) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=kid or f"{kind.value}:{approach}:{offset}",
        keyframe_type=kind,
        frame_ref="object:fruit",
        anchor=anchor,
        approach_axis_xyz=approach,
        tool_axis_to_align=tool,  # type: ignore[arg-type]
        offset_along_approach_m=offset,
        planner=KeyframePlannerType.CARTESIAN,
    )


def _artifact(*keyframes: RelativeKeyframeSpec) -> KeyframePlanArtifact:
    return KeyframePlanArtifact(
        artifact_id="artifact",
        provenance=ArtifactProvenance(
            artifact_id="artifact",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="test",
        ),
        scene_signature="scene",
        subgoal_id="sg",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="side",
                keyframes=list(keyframes),
                rationale="fixture",
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="fixture",
                    input_hash="input",
                ),
            )
        ],
    )


def test_lateral_tabletop_approach_is_rewritten_top_down() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.1,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    pre, grasp, lift = bound.candidates[0].keyframes
    assert grasp.approach_axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert grasp.tool_axis_to_align == "-z"
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert grasp.metadata["contact_geometry_source"] == MULTI_FINGER_TABLETOP_ENCLOSURE
    assert pre.approach_axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert pre.offset_along_approach_m == pytest.approx(0.12)
    assert lift.offset_along_approach_m == pytest.approx(0.18)
    assert bound.candidates[0].metadata["geometry_binder"] == (
        MULTI_FINGER_TABLETOP_ENCLOSURE
    )


def test_thin_tabletop_object_raises_grasp_above_support() -> None:
    """Spoon-scale objects need GRASP TCP above support so 3F tips clear."""

    request = _request(object_z=0.925, dims=(0.045, 0.137, 0.024))
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.1,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    assert grasp.approach_axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert grasp.offset_along_approach_m == pytest.approx(0.03, abs=1e-3)
    assert grasp.metadata["contact_geometry_source"] == MULTI_FINGER_TABLETOP_ENCLOSURE
    assert grasp.metadata["tabletop_enclosure_grasp_clearance_above_support_m"] == (
        pytest.approx(0.03, abs=1e-3)
    )


def test_thick_tabletop_object_keeps_zero_grasp_offset_fruit_regression() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.1,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert "tabletop_enclosure_grasp_clearance_above_support_m" not in grasp.metadata


def test_inverted_tool_axis_topdown_is_corrected() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="+z",
            offset=0.12,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="+z",
            offset=0.0,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    grasp = bound.candidates[0].keyframes[1]
    assert grasp.tool_axis_to_align == "-z"
    assert grasp.approach_axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert grasp.offset_along_approach_m == pytest.approx(0.0)


def test_already_top_down_enclosure_is_left_intact() -> None:
    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.15,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.0,
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.2,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    pre, grasp, lift = bound.candidates[0].keyframes
    assert pre.offset_along_approach_m == pytest.approx(0.15)
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert lift.offset_along_approach_m == pytest.approx(0.2)
    assert grasp.approach_axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert grasp.tool_axis_to_align == "-z"
    assert grasp.metadata["contact_geometry_source"] == MULTI_FINGER_TABLETOP_ENCLOSURE
    assert "tabletop_enclosure_rewritten" not in grasp.metadata


def test_pre_grasp_standoff_candidates_prefer_then_shorten() -> None:
    assert multi_finger_tabletop_pre_grasp_standoff_candidates(0.12) == (
        0.12,
        0.10,
        0.08,
        0.06,
    )
    assert multi_finger_tabletop_pre_grasp_standoff_candidates(0.08) == (
        0.08,
        0.06,
    )
    assert multi_finger_tabletop_pre_grasp_standoff_candidates(0.05) == (0.05,)


def test_lift_standoff_candidates_prefer_then_shorten() -> None:
    from tuj.m5_motion.grasp_geometry import (
        multi_finger_tabletop_lift_standoff_candidates,
    )

    assert multi_finger_tabletop_lift_standoff_candidates(0.18) == (
        0.18,
        0.15,
        0.12,
        0.10,
        0.08,
        0.06,
    )
    assert multi_finger_tabletop_lift_standoff_candidates(0.10) == (
        0.10,
        0.08,
        0.06,
    )

def test_binder_still_emits_preferred_pre_clearance_for_fruit_b_regression() -> None:
    """fruit_b live PASS used preferred 0.12 m PRE; binder must keep emitting it."""

    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.1,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    pre = bound.candidates[0].keyframes[0]
    assert pre.offset_along_approach_m == pytest.approx(0.12)
    assert pre.metadata["contact_geometry_source"] == MULTI_FINGER_TABLETOP_ENCLOSURE


def test_compiler_adapts_enclosure_lift_standoff_when_preferred_unreachable() -> None:
    from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
    from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
    from tuj.m5_motion.strategy import EdgePlanResult

    class _ReachLimitedKinematics:
        def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
            z = float(world_pos[2])
            # Preferred PRE 0.12 (~z>=1.07) and LIFT 0.18 (~z>=1.13) fail;
            # shorter standoffs and GRASP remain reachable.
            if z >= 1.07:
                return IKSolutionSet(
                    attempted_seeds=4,
                    failure_code="IK_NUMERICAL_NON_CONVERGENCE",
                    detail="no full-pose IK solution converged from deterministic seeds",
                )
            q = (0.1, -1.0, 1.2, -1.0, -1.5, 0.2)
            return IKSolutionSet(
                solutions=(
                    IKResult(
                        solved=True,
                        qpos=q,
                        branch_id=f"z{z:.3f}",
                        position_error_m=0.0,
                        orientation_error_rad=0.0,
                    ),
                ),
                best_position_error_m=0.0,
                best_orientation_error_rad=0.0,
                attempted_seeds=4,
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

    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.12,
            kid="pre",
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.0,
            kid="grasp",
        ),
        _keyframe(
            kind=KeyframeType.LIFT,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.18,
            kid="lift",
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    keyframes = []
    for kf in bound.candidates[0].keyframes:
        keyframes.append(
            kf.model_copy(
                update={
                    "metadata": {
                        **kf.metadata,
                        "contact_geometry_source": MULTI_FINGER_TABLETOP_ENCLOSURE,
                    }
                }
            )
        )
    candidate = bound.candidates[0].model_copy(update={"keyframes": keyframes})
    result = FirstFeasibleStrategyCompiler(_ReachLimitedKinematics()).compile(
        request.world,
        [candidate],
        start_joint_config=(0.0, -1.2, 1.4, -1.5, -1.5, 0.0),
        state_validator=_AlwaysValid(),
        edge_planner=_AlwaysEdge(),
    )
    assert result.solved
    assert result.connected is not None
    pre, grasp, lift = [node.keyframe for node in result.connected.nodes]
    assert pre.offset_along_approach_m == pytest.approx(0.10)
    assert grasp.offset_along_approach_m == pytest.approx(0.0)
    assert lift.offset_along_approach_m == pytest.approx(0.10)
    assert lift.metadata["tabletop_enclosure_standoff_adapted_m"] == pytest.approx(0.10)
    assert lift.metadata["tabletop_enclosure_standoff_preferred_m"] == pytest.approx(
        0.18
    )


def test_compiler_adapts_enclosure_pre_standoff_when_preferred_unreachable() -> None:
    """Outer-workspace PRE@0.12 can be IK-empty while shorter PRE remains valid."""

    from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
    from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
    from tuj.m5_motion.strategy import EdgePlanResult

    class _ReachLimitedKinematics:
        def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
            z = float(world_pos[2])
            if z >= 1.07:
                return IKSolutionSet(
                    attempted_seeds=4,
                    failure_code="IK_NUMERICAL_NON_CONVERGENCE",
                    detail="no full-pose IK solution converged from deterministic seeds",
                )
            q = (0.1, -1.0, 1.2, -1.0, -1.5, 0.2)
            return IKSolutionSet(
                solutions=(
                    IKResult(
                        solved=True,
                        qpos=q,
                        branch_id=f"z{z:.3f}",
                        position_error_m=0.0,
                        orientation_error_rad=0.0,
                    ),
                ),
                best_position_error_m=0.0,
                best_orientation_error_rad=0.0,
                attempted_seeds=4,
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

    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.12,
            kid="pre",
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.0,
            kid="grasp",
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    pre = bound.candidates[0].keyframes[0]
    assert pre.offset_along_approach_m == pytest.approx(0.12)
    pre = pre.model_copy(
        update={
            "metadata": {
                **pre.metadata,
                "contact_geometry_source": MULTI_FINGER_TABLETOP_ENCLOSURE,
            }
        }
    )
    candidate = bound.candidates[0].model_copy(
        update={"keyframes": [pre, bound.candidates[0].keyframes[1]]}
    )

    result = FirstFeasibleStrategyCompiler(_ReachLimitedKinematics()).compile(
        request.world,
        [candidate],
        start_joint_config=(0.0, -1.2, 1.4, -1.5, -1.5, 0.0),
        state_validator=_AlwaysValid(),
        edge_planner=_AlwaysEdge(),
    )
    assert result.solved
    assert result.connected is not None
    adapted_pre = result.connected.nodes[0].keyframe
    assert adapted_pre.offset_along_approach_m == pytest.approx(0.10)
    assert adapted_pre.metadata["tabletop_enclosure_pre_standoff_adapted_m"] == pytest.approx(
        0.10
    )
    assert adapted_pre.metadata[
        "tabletop_enclosure_pre_standoff_preferred_m"
    ] == pytest.approx(0.12)


def test_compiler_keeps_preferred_pre_when_reachable_fruit_b_regression() -> None:
    from tuj.m5_motion.compiler import FirstFeasibleStrategyCompiler
    from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
    from tuj.m5_motion.strategy import EdgePlanResult

    class _AlwaysIk:
        def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
            q = (0.1, -1.0, 1.2, -1.0, -1.5, 0.2)
            return IKSolutionSet(
                solutions=(
                    IKResult(
                        solved=True,
                        qpos=q,
                        branch_id="ok",
                        position_error_m=0.0,
                        orientation_error_rad=0.0,
                    ),
                ),
                best_position_error_m=0.0,
                best_orientation_error_rad=0.0,
                attempted_seeds=4,
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

    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.12,
            kid="pre",
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.0,
            kid="grasp",
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    pre = bound.candidates[0].keyframes[0].model_copy(
        update={
            "metadata": {
                **bound.candidates[0].keyframes[0].metadata,
                "contact_geometry_source": MULTI_FINGER_TABLETOP_ENCLOSURE,
            }
        }
    )
    candidate = bound.candidates[0].model_copy(
        update={"keyframes": [pre, bound.candidates[0].keyframes[1]]}
    )
    result = FirstFeasibleStrategyCompiler(_AlwaysIk()).compile(
        request.world,
        [candidate],
        start_joint_config=(0.0, -1.2, 1.4, -1.5, -1.5, 0.0),
        state_validator=_AlwaysValid(),
        edge_planner=_AlwaysEdge(),
    )
    assert result.solved
    assert result.connected is not None
    kept = result.connected.nodes[0].keyframe
    assert kept.offset_along_approach_m == pytest.approx(0.12)
    assert "tabletop_enclosure_pre_standoff_adapted_m" not in kept.metadata


def test_vacuum_acquire_is_not_rewritten_by_tabletop_binder() -> None:
    request = _request(ee="vac", capabilities=["suction"])
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    assert bound.artifact_id == artifact.artifact_id
    assert bound.candidates[0].keyframes[1].approach_axis_xyz == pytest.approx(
        (0.0, -1.0, 0.0)
    )
    suction = bind_suction_surface_contact(artifact, request)
    assert suction.candidates[0].keyframes[1].approach_axis_xyz == pytest.approx(
        (0.0, -1.0, 0.0)
    )


def test_two_finger_opposed_provider_skips_tabletop_binder() -> None:
    request = _request(ee="2F", capabilities=["opposed_finger_contact"])
    request.task.metadata["grasp_geometry_provider"] = "TWO_FINGER_OPPOSED_CONTACT"
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    assert bound.candidates[0].keyframes[1].approach_axis_xyz == pytest.approx(
        (0.0, -1.0, 0.0)
    )


def test_free_space_object_without_support_block_is_unchanged() -> None:
    request = _request(object_z=1.5)
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.05,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, -1.0, 0.0),
            tool="-z",
            offset=0.0,
        ),
    )
    bound = bind_multi_finger_tabletop_enclosure(artifact, request)
    assert bound.artifact_id == artifact.artifact_id

def _place_request_with_attachment(
    *,
    object_z: float = 0.928,
    region_z: float = 0.949,
    region_half_z: float = 0.029,
    grasp_z_in_ref: float = 0.027,
) -> MotionPlanRequest:
    from tuj.m5_motion.schema import AttachedObjectTransform, MotionConstraints, Pose

    half = region_half_z
    return MotionPlanRequest(
        request_id="request:mf-place",
        provenance=ArtifactProvenance(
            artifact_id="request-artifact",
            artifact_type="MotionPlanRequest",
            produced_by=ModuleName.TASK_PLANNER,
            invocation_id="test",
        ),
        world=WorldSnapshot(
            scene=SceneRef(signature="scene"),
            robot_state=RobotState(
                robot_id="robot",
                joint_names=["j1"],
                joint_positions_rad=[0.0],
                attached_object_id="utensil",
                eef_pose=Pose(
                    frame_id="world",
                    position_m=(2.2, -2.8, 1.1),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            objects={
                "utensil": {
                    "pose": {
                        "frame_id": "world",
                        "position_m": [2.4, -3.4, object_z],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.03, 0.15, 0.02],
                    "anchors": {
                        "center": [0.0, 0.0, 0.0],
                        "top": [0.0, 0.0, 0.01],
                        "bottom": [0.0, 0.0, -0.01],
                    },
                },
                "tray": {
                    "pose": {
                        "frame_id": "world",
                        "position_m": [2.3, -3.1, region_z],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "dimensions_m": [0.4, 0.5, 2 * half],
                    "anchors": {
                        "center": [0.0, 0.0, 0.0],
                        "top": [0.0, 0.0, half],
                        "bottom": [0.0, 0.0, -half],
                    },
                    "collision_points_m": [
                        [0.0, 0.0, -half + 0.015],
                        [0.05, 0.05, -half + 0.015],
                        [-0.05, 0.05, -half + 0.015],
                        [0.05, -0.05, -half + 0.015],
                        [-0.05, -0.05, -half + 0.015],
                        [0.0, 0.0, -half],
                        [0.02, 0.02, -half],
                        [-0.02, -0.02, -half],
                    ],
                },
            },
            metadata={
                "attached_object_transforms": {
                    "utensil": AttachedObjectTransform(
                        object_id="utensil",
                        free_joint_name="utensil_joint0",
                        reference_kind="site",
                        reference_name="grip",
                        position_in_reference_m=(0.0, 0.0, grasp_z_in_ref),
                        orientation_in_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ).model_dump(mode="json")
                }
            },
        ),
        task=MotionTask(
            task_id="place-utensil",
            subgoal_id="sg",
            action_type="place",
            ee="3F",
            target_ids=["utensil"],
            goal=MotionGoal(
                goal_type=GoalType.POSE,
                target_object_id="utensil",
                target_region_id="tray",
                target_pose=Pose(
                    frame_id="world",
                    position_m=(2.4, -3.4, object_z),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            metadata={
                "operation": "PLACE",
                "ee_capabilities": ["multi_finger_contact"],
            },
        ),
        constraints=MotionConstraints(collision_margin_m=0.005),
    )


def test_multi_finger_place_raises_release_above_region_floor() -> None:
    from tuj.m5_motion.attachment_retarget import (
        ATTACHED_OBJECT_POSE_SUBJECT,
        POSE_SUBJECT_KEY,
        POSE_SUBJECT_OBJECT_ID_KEY,
        retarget_resolved_pose,
    )
    from tuj.m5_motion.geometry import RelativePoseResolver
    from tuj.m5_motion.grasp_geometry import (
        MULTI_FINGER_PLACE_SUPPORT_CLEARANCE,
        bind_multi_finger_place_support_clearance,
        multi_finger_finger_below_tcp_m,
    )

    request = _place_request_with_attachment()
    artifact = KeyframePlanArtifact(
        artifact_id="artifact",
        provenance=ArtifactProvenance(
            artifact_id="artifact",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="test",
        ),
        scene_signature="scene",
        subgoal_id="sg",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="place",
                keyframes=[
                    RelativeKeyframeSpec(
                        keyframe_id="pre",
                        keyframe_type=KeyframeType.PRE_PLACE,
                        frame_ref="object:utensil",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.12,
                        planner=KeyframePlannerType.CARTESIAN,
                        metadata={
                            POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
                            POSE_SUBJECT_OBJECT_ID_KEY: "utensil",
                            "packing_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    ),
                    RelativeKeyframeSpec(
                        keyframe_id="place",
                        keyframe_type=KeyframeType.PLACE,
                        frame_ref="object:utensil",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.0,
                        planner=KeyframePlannerType.CARTESIAN,
                        metadata={
                            POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
                            POSE_SUBJECT_OBJECT_ID_KEY: "utensil",
                            "packing_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    ),
                ],
                rationale="fixture",
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="fixture",
                    input_hash="input",
                ),
            )
        ],
    )
    bound = bind_multi_finger_place_support_clearance(artifact, request)
    place = bound.candidates[0].keyframes[1]
    assert place.offset_along_approach_m > 0.0
    assert place.metadata["contact_geometry_source"] == MULTI_FINGER_PLACE_SUPPORT_CLEARANCE
    # Rimmed trays: clearance plane is region top (not only interior floor).
    region_top_z = 0.949 + 0.029
    min_tcp = (
        region_top_z
        + multi_finger_finger_below_tcp_m()
        + float(request.constraints.collision_margin_m)
    )
    eef = retarget_resolved_pose(
        request.world, place, RelativePoseResolver(request.world).resolve(place)
    )
    assert eef.position_m[2] >= min_tcp - 1e-9


def test_multi_finger_place_binder_skips_acquire() -> None:
    from tuj.m5_motion.grasp_geometry import bind_multi_finger_place_support_clearance

    request = _request()
    artifact = _artifact(
        _keyframe(
            kind=KeyframeType.PRE_GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.12,
        ),
        _keyframe(
            kind=KeyframeType.GRASP,
            approach=(0.0, 0.0, 1.0),
            tool="-z",
            offset=0.0,
        ),
    )
    bound = bind_multi_finger_place_support_clearance(artifact, request)
    assert bound.artifact_id == artifact.artifact_id


def test_acquire_world_frame_lift_rewritten_onto_object() -> None:
    """PRE/GRASP/LIFT on frame_ref=world resolve at the origin (~unreachable)."""

    from tuj.m5_motion.geometry import RelativePoseResolver
    from tuj.m5_motion.grasp_geometry import (
        ACQUIRE_WORLD_CONTACT_FRAME,
        bind_acquire_world_contact_frames,
    )

    request = _request()
    artifact = KeyframePlanArtifact(
        artifact_id="artifact",
        provenance=ArtifactProvenance(
            artifact_id="artifact",
            artifact_type="KeyframePlanArtifact",
            produced_by=ModuleName.MOTION_PLANNER,
            invocation_id="test",
        ),
        scene_signature="scene",
        subgoal_id="sg",
        candidates=[
            KeyframePlanCandidate(
                strategy_id="strategy1",
                keyframes=[
                    RelativeKeyframeSpec(
                        keyframe_id="k0",
                        keyframe_type=KeyframeType.PRE_GRASP,
                        frame_ref="object:fruit",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.12,
                        planner=KeyframePlannerType.CARTESIAN,
                    ),
                    RelativeKeyframeSpec(
                        keyframe_id="k1",
                        keyframe_type=KeyframeType.GRASP,
                        frame_ref="object:fruit",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.0,
                        planner=KeyframePlannerType.CARTESIAN,
                    ),
                    RelativeKeyframeSpec(
                        keyframe_id="k3",
                        keyframe_type=KeyframeType.LIFT,
                        frame_ref="world",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.18,
                        planner=KeyframePlannerType.CARTESIAN,
                    ),
                ],
                rationale="fixture",
                provenance=StrategyGenerationProvenance(
                    generator_kind=StrategyGeneratorKind.TEMPLATE,
                    generator_id="fixture",
                    input_hash="input",
                ),
            )
        ],
    )
    resolver = RelativePoseResolver(request.world)
    raw_lift = resolver.resolve(artifact.candidates[0].keyframes[2])
    assert raw_lift.position_m == pytest.approx((0.0, 0.0, 0.18))

    bound = bind_acquire_world_contact_frames(artifact, request)
    lift = bound.candidates[0].keyframes[2]
    assert lift.frame_ref == "object:fruit"
    assert lift.metadata["contact_geometry_source"] == ACQUIRE_WORLD_CONTACT_FRAME
    fixed = resolver.resolve(lift)
    fruit_z = float(request.world.objects["fruit"]["pose"]["position_m"][2])
    assert fixed.position_m[2] == pytest.approx(fruit_z + 0.18)
    assert fixed.position_m[0] != pytest.approx(0.0)


def test_spoon_a_world_lift_vs_spoon_b_object_lift_reach_regression() -> None:
    """Offline: failed spoon_a world-LIFT vs successful spoon_b object-LIFT.

    Live SG1_s4_d1 failed all strategies at k3 with ~3.747 m from base because
    LIFT used frame_ref=world → resolved TCP ≈ (0,0,0.18). spoon_b acquire kept
    object-frame contact phases. Same rewrite must fix spoon_a without moving
    already-correct spoon_b frames.
    """

    from pathlib import Path

    import numpy as np

    from tuj.m5_motion.geometry import RelativePoseResolver
    from tuj.m5_motion.grasp_geometry import (
        bind_acquire_world_contact_frames,
        bind_multi_finger_tabletop_enclosure,
    )
    from tuj.m5_motion.schema import MotionPlanRequest

    repo = Path(__file__).resolve().parents[4]
    spoon_a_path = repo / (
        "output/c3_2/m5/requests/0030-motion-request_0_19bbb12f5492ed15b92a.json"
    )
    spoon_b_path = repo / (
        "output/c3_2/m5/requests/0021-motion-request_0_19c66bcdb156f6cc5c44.json"
    )
    if not spoon_a_path.is_file() or not spoon_b_path.is_file():
        pytest.skip("c3_2 spoon acquire request artifacts not present")

    def _strategy(oid: str, *, lift_frame: str) -> KeyframePlanArtifact:
        def kf(kid, kind, off, frame):
            return RelativeKeyframeSpec(
                keyframe_id=kid,
                keyframe_type=kind,
                frame_ref=frame,
                anchor="center",
                approach_axis_xyz=(0.0, 0.0, 1.0),
                tool_axis_to_align="-z",
                offset_along_approach_m=off,
                planner=KeyframePlannerType.CARTESIAN,
            )

        return KeyframePlanArtifact(
            artifact_id=f"artifact:{oid}",
            provenance=ArtifactProvenance(
                artifact_id=f"artifact:{oid}",
                artifact_type="KeyframePlanArtifact",
                produced_by=ModuleName.MOTION_PLANNER,
                invocation_id="test",
            ),
            scene_signature="scene",
            subgoal_id="sg",
            candidates=[
                KeyframePlanCandidate(
                    strategy_id="strategy1",
                    keyframes=[
                        kf("k0", KeyframeType.PRE_GRASP, 0.05, f"object:{oid}"),
                        kf("k1", KeyframeType.PRE_GRASP, 0.08, f"object:{oid}"),
                        kf("k2", KeyframeType.GRASP, 0.0, f"object:{oid}"),
                        kf("k3", KeyframeType.LIFT, 0.18, lift_frame),
                    ],
                    rationale="fixture",
                    provenance=StrategyGenerationProvenance(
                        generator_kind=StrategyGeneratorKind.TEMPLATE,
                        generator_id="fixture",
                        input_hash="input",
                    ),
                )
            ],
        )

    def _base_xy_distance(req: MotionPlanRequest, pose_xyz) -> float:
        base = np.asarray(
            req.world.metadata["robot_base_world_m"], dtype=float
        )
        return float(np.linalg.norm(np.asarray(pose_xyz, float) - base))

    spoon_a = MotionPlanRequest.model_validate_json(
        spoon_a_path.read_text(encoding="utf-8")
    )
    spoon_b = MotionPlanRequest.model_validate_json(
        spoon_b_path.read_text(encoding="utf-8")
    )

    # Failed pattern: LIFT on world.
    raw_a = _strategy("spoon_a", lift_frame="world")
    resolver_a = RelativePoseResolver(spoon_a.world)
    raw_lift_a = resolver_a.resolve(raw_a.candidates[0].keyframes[3])
    assert raw_lift_a.position_m[0] == pytest.approx(0.0)
    assert _base_xy_distance(spoon_a, raw_lift_a.position_m) > 3.0

    bound_a = bind_multi_finger_tabletop_enclosure(
        bind_acquire_world_contact_frames(raw_a, spoon_a), spoon_a
    )
    lift_a = bound_a.candidates[0].keyframes[3]
    assert lift_a.frame_ref == "object:spoon_a"
    fixed_a = resolver_a.resolve(lift_a)
    assert _base_xy_distance(spoon_a, fixed_a.position_m) < 1.2

    # Successful pattern: LIFT already on object — rewrite is a no-op for frame.
    raw_b = _strategy("spoon_b", lift_frame="object:spoon_b")
    bound_b = bind_multi_finger_tabletop_enclosure(
        bind_acquire_world_contact_frames(raw_b, spoon_b), spoon_b
    )
    lift_b = bound_b.candidates[0].keyframes[3]
    assert lift_b.frame_ref == "object:spoon_b"
    fixed_b = RelativePoseResolver(spoon_b.world).resolve(lift_b)
    assert _base_xy_distance(spoon_b, fixed_b.position_m) < 1.2
