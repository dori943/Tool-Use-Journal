from __future__ import annotations

import pytest

from tuj.m5_motion.attachment_retarget import (
    ATTACHED_OBJECT_POSE_SUBJECT,
    POSE_SUBJECT_KEY,
    POSE_SUBJECT_OBJECT_ID_KEY,
)
from tuj.m5_motion.kinematics import IKResult, IKSolutionSet
from tuj.m5_motion.mujoco_collision import CollisionCheckResult
from tuj.m5_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m5_motion.schema import (
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m5_motion.strategy import EdgePlanResult


def _keyframe(planner: KeyframePlannerType) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id=f"target-{planner.value}",
        keyframe_type=KeyframeType.CUSTOM,
        frame_ref="object:target",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=planner,
    )


class _PlanarKinematics:
    def __init__(self):
        self.seeds = []

    def forward_pose_world(self, qpos):
        return (qpos[0], qpos[1], 0.0), (0.0, 0.0, 0.0, 1.0)

    def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
        del orientation_xyzw
        self.seeds.append(kwargs.get("seed_qpos"))
        return IKSolutionSet(
            solutions=(
                IKResult(
                    solved=True,
                    qpos=(float(world_pos[0]), float(world_pos[1])),
                    position_error_m=0.0,
                    orientation_error_rad=0.0,
                    branch_id="planar",
                ),
            ),
            enumeration_complete=True,
        )


def test_cartesian_planner_follows_straight_pose_samples() -> None:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="planar",
            joint_names=["x", "y"],
            joint_positions_rad=[0.0, 0.0],
        ),
        objects={
            "target": {
                "pose": {
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    kinematics = _PlanarKinematics()
    planner = CartesianEdgePlanner(
        kinematics,
        world,
        lambda q, keyframe: abs(q[1]) < 1e-9,
        translation_step_m=0.25,
        max_joint_step_rad=0.1,
    )

    result = planner.plan(
        (0.0, 0.0),
        (1.0, 0.0),
        None,
        _keyframe(KeyframePlannerType.CARTESIAN),
    )

    assert result.valid
    assert result.joint_path[0] == (0.0, 0.0)
    assert result.joint_path[-1] == (1.0, 0.0)
    assert all(abs(q[1]) < 1e-9 for q in result.joint_path)
    assert len(result.joint_path) > 4
    assert kinematics.seeds
    assert kinematics.seeds[0] == (0.0, 0.0)
    assert all(seed is not None for seed in kinematics.seeds)


class _ContinuationSensitiveKinematics:
    """Expose the unsafe global-seed jump that local Cartesian IK must avoid."""

    def forward_pose_world(self, qpos):
        return (qpos[0], 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
        del orientation_xyzw
        seed = kwargs.get("seed_qpos")
        nearby = float(world_pos[0]) if seed is not None else float(world_pos[0]) + 4.0
        return IKSolutionSet(
            solutions=(
                IKResult(
                    solved=True,
                    qpos=(nearby,),
                    position_error_m=0.0,
                    orientation_error_rad=0.0,
                    branch_id="same-coarse-branch",
                ),
            ),
            enumeration_complete=True,
        )


def test_cartesian_planner_seeds_each_intermediate_ik_from_previous_state() -> None:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="continuation",
            joint_names=["x"],
            joint_positions_rad=[0.0],
        ),
        objects={
            "target": {
                "pose": {
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    planner = CartesianEdgePlanner(
        _ContinuationSensitiveKinematics(),
        world,
        lambda q, keyframe: True,
        translation_step_m=0.25,
        max_joint_step_rad=0.1,
        wrap_joints=False,
    )

    result = planner.plan(
        (0.0,),
        (1.0,),
        None,
        _keyframe(KeyframePlannerType.CARTESIAN),
    )

    assert result.valid
    assert result.joint_path[0] == (0.0,)
    assert result.joint_path[-1] == (1.0,)
    assert max(q[0] for q in result.joint_path) <= 1.0


def test_cartesian_samples_retarget_attached_object_pose_to_eef_pose() -> None:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="planar",
            joint_names=["x", "y"],
            joint_positions_rad=[0.0, 0.0],
            attached_object_id="target",
        ),
        objects={
            "target": {
                "pose": {
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
        metadata={
            "attached_object_transforms": {
                "target": {
                    "object_id": "target",
                    "free_joint_name": "target_free",
                    "reference_kind": "site",
                    "reference_name": "grip_site",
                    "position_in_reference_m": [0.1, 0.0, 0.0],
                    "orientation_in_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    keyframe = _keyframe(KeyframePlannerType.CARTESIAN)
    keyframe.metadata = {
        POSE_SUBJECT_KEY: ATTACHED_OBJECT_POSE_SUBJECT,
        POSE_SUBJECT_OBJECT_ID_KEY: "target",
    }
    planner = CartesianEdgePlanner(
        _PlanarKinematics(),
        world,
        lambda q, supplied: True,
        translation_step_m=0.25,
        max_joint_step_rad=0.1,
    )

    result = planner.plan((0.0, 0.0), (0.9, 0.0), None, keyframe)

    assert result.valid
    eef_x = [q[0] for q in result.joint_path]
    assert eef_x[-1] == pytest.approx(0.9)
    assert max(eef_x) <= 0.9
    assert any(value == pytest.approx(0.225) for value in eef_x)


class _NonConvergingKinematics:
    def forward_pose_world(self, qpos):
        return (qpos[0], 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
        del world_pos, orientation_xyzw, kwargs
        return IKSolutionSet(
            attempted_seeds=13,
            best_position_error_m=0.031,
            best_orientation_error_rad=0.22,
            failure_code="IK_NUMERICAL_NON_CONVERGENCE",
            detail="iteration budget exhausted",
            enumeration_complete=True,
        )


def test_cartesian_failure_reports_numerical_ik_diagnostics() -> None:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="non-converging",
            joint_names=["x"],
            joint_positions_rad=[0.0],
        ),
        objects={
            "target": {
                "pose": {
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    planner = CartesianEdgePlanner(
        _NonConvergingKinematics(),
        world,
        lambda q, keyframe: True,
        translation_step_m=0.25,
    )

    result = planner.plan(
        (0.0,),
        (1.0,),
        None,
        _keyframe(KeyframePlannerType.CARTESIAN),
    )

    assert not result.valid
    assert result.failure_code == "CARTESIAN_INTERMEDIATE_IK_NON_CONVERGENCE"
    assert "sample 1/4" in result.detail
    assert "attempted_seeds=13" in result.detail
    assert "iteration budget exhausted" in result.detail


class _CollisionRejectingValidator:
    def check(self, config, keyframe):
        del config, keyframe
        return CollisionCheckResult(
            valid=False,
            failure_code="COLLISION_MARGIN_VIOLATION",
            detail="forearm <-> island",
            min_clearance_m=-0.01,
        )


def test_cartesian_failure_distinguishes_valid_ik_from_invalid_state() -> None:
    world = WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="planar",
            joint_names=["x", "y"],
            joint_positions_rad=[0.0, 0.0],
        ),
        objects={
            "target": {
                "pose": {
                    "position_m": [1.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        },
    )
    planner = CartesianEdgePlanner(
        _PlanarKinematics(),
        world,
        _CollisionRejectingValidator(),
        translation_step_m=0.25,
    )

    result = planner.plan(
        (0.0, 0.0),
        (1.0, 0.0),
        None,
        _keyframe(KeyframePlannerType.CARTESIAN),
    )

    assert not result.valid
    assert result.failure_code == "CARTESIAN_INTERMEDIATE_STATE_INVALID"
    assert "COLLISION_MARGIN_VIOLATION" in result.detail
    assert "forearm <-> island" in result.detail
    assert result.min_clearance_m == -0.01


def test_rrt_connect_routes_around_invalid_joint_region() -> None:
    def valid(q, keyframe):
        del keyframe
        return not (abs(q[0]) < 0.25 and abs(q[1]) < 0.25)

    planner = RRTConnectEdgePlanner(
        valid,
        joint_limits_rad=((-1.0, 1.0), (-1.0, 1.0)),
        random_seed=11,
        max_iterations=3000,
        timeout_s=2.0,
        extension_step_rad=0.15,
        validation_step_rad=0.02,
    )

    result = planner.plan(
        (-0.9, 0.0),
        (0.9, 0.0),
        None,
        _keyframe(KeyframePlannerType.SAMPLING_BASED),
    )

    assert result.valid
    assert result.joint_path[0] == (-0.9, 0.0)
    assert result.joint_path[-1] == (0.9, 0.0)
    assert any(abs(q[1]) >= 0.25 for q in result.joint_path)
    assert all(valid(q, None) for q in result.joint_path)


class _TaggedPlanner:
    def __init__(self, tag: float) -> None:
        self.tag = tag

    def plan(self, source, target, source_keyframe, target_keyframe):
        del source_keyframe, target_keyframe
        return EdgePlanResult(
            valid=True,
            joint_path=(tuple(source), (self.tag,), tuple(target)),
        )


def test_dispatch_uses_keyframe_planner_type() -> None:
    dispatcher = PlannerDispatchEdgePlanner(
        joint=_TaggedPlanner(1.0),
        cartesian=_TaggedPlanner(2.0),
        sampling_based=_TaggedPlanner(3.0),
    )

    result = dispatcher.plan(
        (0.0,),
        (4.0,),
        None,
        _keyframe(KeyframePlannerType.SAMPLING_BASED),
    )

    assert result.joint_path[1] == (3.0,)
