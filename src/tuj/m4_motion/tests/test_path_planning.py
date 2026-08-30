from __future__ import annotations

from tuj.m4_motion.kinematics import IKResult, IKSolutionSet
from tuj.m4_motion.path_planning import (
    CartesianEdgePlanner,
    PlannerDispatchEdgePlanner,
    RRTConnectEdgePlanner,
)
from tuj.m4_motion.schema import (
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    RobotState,
    SceneRef,
    WorldSnapshot,
)
from tuj.m4_motion.strategy import EdgePlanResult


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
    def forward_pose_world(self, qpos):
        return (qpos[0], qpos[1], 0.0), (0.0, 0.0, 0.0, 1.0)

    def solve_all_ik(self, world_pos, orientation_xyzw, **kwargs):
        del orientation_xyzw, kwargs
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
    planner = CartesianEdgePlanner(
        _PlanarKinematics(),
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
