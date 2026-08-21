from __future__ import annotations

import mujoco

from motion_planner.mujoco_collision import (
    MuJoCoCollisionModelRegistry,
    MuJoCoCollisionValidator,
    MuJoCoInterpolatingEdgePlanner,
)
from motion_planner.schema import (
    AttachedObjectTransform,
    CollisionContext,
    FreeObjectPose,
    KeyframePlannerType,
    KeyframeType,
    RelativeKeyframeSpec,
    Pose,
    TrajectoryWaypoint,
)


_SCENE = """
<mujoco model="collision-test">
  <compiler autolimits="true"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root" pos="0 0 0.2">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="robot_col" type="sphere" size="0.1"
            contype="1" conaffinity="1"/>
    </body>
    <body name="obstacle">
      <geom name="wall_col" type="sphere" pos="0.5 0 0.2" size="0.1"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_ATTACHMENT_SCENE = """
<mujoco model="attachment-test">
  <compiler autolimits="true"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="robot_root">
      <joint name="arm_slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="hand_col" type="sphere" size="0.04"
            contype="1" conaffinity="1"/>
      <body name="finger">
        <joint name="finger_slide" type="slide" axis="0 1 0" range="0 0.3"/>
        <geom name="finger_col" type="sphere" size="0.04"
              contype="1" conaffinity="1"/>
      </body>
    </body>
    <body name="part">
      <freejoint name="part_free"/>
      <geom name="part_col" type="sphere" size="0.05"
            contype="1" conaffinity="1"/>
    </body>
    <body name="carry_obstacle">
      <geom name="carry_wall_col" type="sphere" pos="0.65 0 0" size="0.05"
            contype="1" conaffinity="1"/>
    </body>
    <body name="finger_obstacle">
      <geom name="finger_wall_col" type="sphere" pos="0 0.28 0" size="0.04"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _keyframe(context_id: str | None = None) -> RelativeKeyframeSpec:
    return RelativeKeyframeSpec(
        keyframe_id="target",
        keyframe_type=KeyframeType.CUSTOM,
        frame_ref="object:target",
        anchor="center",
        approach_axis_xyz=(0.0, 0.0, 1.0),
        planner=KeyframePlannerType.JOINT,
        collision_context_id=context_id,
    )


def _validator(
    *,
    contexts: dict[str, CollisionContext] | None = None,
    model_version: str = "test-v1",
) -> MuJoCoCollisionValidator:
    model = mujoco.MjModel.from_xml_string(_SCENE)
    return MuJoCoCollisionValidator(
        model,
        joint_names=("slide",),
        robot_root_body_name="robot_root",
        collision_margin_m=0.05,
        collision_model_version=model_version,
        collision_contexts=contexts,
        entity_geoms={"wall": ("wall_col",)},
    )


def test_endpoint_validator_checks_clearance_and_joint_limits() -> None:
    validator = _validator()

    safe = validator.check((0.0,), _keyframe())
    collision = validator.check((0.3,), _keyframe())
    outside_limit = validator.check((1.1,), _keyframe())

    assert safe.valid
    assert safe.clearance_is_lower_bound
    assert safe.min_clearance_m == 0.05
    assert not collision.valid
    assert collision.failure_code == "COLLISION_MARGIN_VIOLATION"
    assert collision.min_clearance_m == 0.0
    assert {collision.contacts[0].geom_a, collision.contacts[0].geom_b} == {
        "robot_col",
        "wall_col",
    }
    assert not outside_limit.valid
    assert outside_limit.failure_code == "JOINT_LIMIT_VIOLATION"


def test_context_acm_allows_only_the_declared_pair() -> None:
    touch = CollisionContext(
        context_id="touch",
        allowed_collision_pairs=[("robot", "wall")],
        collision_model_version="test-v1",
    )
    validator = _validator(contexts={touch.context_id: touch})

    result = validator.check((0.3,), _keyframe("touch"))

    assert result.valid
    assert result.contacts
    assert all(contact.allowed for contact in result.contacts)


def test_unknown_collision_context_fails_closed() -> None:
    result = _validator().check((0.0,), _keyframe("not-registered"))

    assert not result.valid
    assert result.failure_code == "COLLISION_CONTEXT_MISSING"


def test_collision_model_version_mismatch_fails_closed() -> None:
    wrong_model = CollisionContext(
        context_id="wrong-model",
        collision_model_version="attached-ee-model-v2",
    )
    validator = _validator(contexts={wrong_model.context_id: wrong_model})

    result = validator.check((0.0,), _keyframe("wrong-model"))

    assert not result.valid
    assert result.failure_code == "COLLISION_MODEL_MISMATCH"


def test_model_registry_routes_event_scoped_collision_model() -> None:
    attached = CollisionContext(
        context_id="attached",
        collision_model_version="attached-v2",
    )
    registry = MuJoCoCollisionModelRegistry(
        {
            "test-v1": _validator(model_version="test-v1"),
            "attached-v2": _validator(model_version="attached-v2"),
        },
        collision_contexts={attached.context_id: attached},
        default_model_version="test-v1",
    )

    result = registry.check((0.0,), _keyframe("attached"))

    assert result.valid
    assert registry.joint_names == ("slide",)


def test_model_registry_fails_when_required_compiled_model_is_absent() -> None:
    attached = CollisionContext(
        context_id="attached",
        collision_model_version="attached-v2",
    )
    registry = MuJoCoCollisionModelRegistry(
        {"test-v1": _validator(model_version="test-v1")},
        collision_contexts={attached.context_id: attached},
        default_model_version="test-v1",
    )

    result = registry.check((0.0,), _keyframe("attached"))

    assert not result.valid
    assert result.failure_code == "COLLISION_MODEL_UNAVAILABLE"


def test_interpolated_edge_rejects_a_collision_between_safe_endpoints() -> None:
    validator = _validator()
    planner = MuJoCoInterpolatingEdgePlanner(
        validator=validator,
        max_joint_step_rad=0.025,
    )
    assert validator.check((0.0,), _keyframe()).valid
    assert validator.check((0.9,), _keyframe()).valid

    edge = planner.plan((0.0,), (0.9,), None, _keyframe())

    assert not edge.valid
    assert edge.failure_code == "COLLISION_MARGIN_VIOLATION"
    assert "invalid sample" in edge.detail


def test_final_waypoint_validation_uses_segment_context() -> None:
    context = CollisionContext(
        context_id="default",
        collision_model_version="test-v1",
    )
    validator = _validator(contexts={context.context_id: context})
    waypoints = (
        TrajectoryWaypoint(time_from_start_s=0.0, joint_positions_rad=[0.0]),
        TrajectoryWaypoint(time_from_start_s=1.0, joint_positions_rad=[0.3]),
    )

    report = validator.check_waypoints(waypoints, context)

    assert not report.valid
    assert report.failed_state_index == 1
    assert report.checked_states == 2
    assert not validator.final_segment_validator(waypoints, context)


def test_attached_free_joint_object_moves_with_reference_body() -> None:
    context = CollisionContext(
        context_id="carrying",
        attached_object_ids=["part"],
        attached_object_transforms=[
            AttachedObjectTransform(
                object_id="part",
                free_joint_name="part_free",
                reference_kind="body",
                reference_name="robot_root",
                position_in_reference_m=(0.2, 0.0, 0.0),
                orientation_in_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        ],
        touch_links=["robot_root", "finger"],
        collision_model_version="attachment-v1",
    )
    validator = MuJoCoCollisionValidator(
        mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE),
        joint_names=("arm_slide",),
        robot_root_body_name="robot_root",
        collision_margin_m=0.02,
        collision_model_version="attachment-v1",
        collision_contexts={context.context_id: context},
        entity_geoms={"part": ("part",)},
    )

    safe = validator.check((0.0,), context=context)
    collision = validator.check((0.35,), context=context)

    assert safe.valid
    assert not collision.valid
    assert collision.failure_code == "COLLISION_MARGIN_VIOLATION"
    assert any(
        {contact.geom_a, contact.geom_b} == {"part_col", "carry_wall_col"}
        for contact in collision.contacts
        if not contact.allowed
    )


def test_detached_free_joint_object_uses_context_world_pose() -> None:
    far = CollisionContext(
        context_id="far",
        free_object_poses=[
            FreeObjectPose(
                object_id="part",
                free_joint_name="part_free",
                pose=Pose(
                    frame_id="world",
                    position_m=(0.35, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        ],
        collision_model_version="attachment-v1",
    )
    near = far.model_copy(
        update={
            "context_id": "near",
            "free_object_poses": [
                FreeObjectPose(
                    object_id="part",
                    free_joint_name="part_free",
                    pose=Pose(
                        frame_id="world",
                        position_m=(0.05, 0.0, 0.0),
                        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    ),
                )
            ],
        }
    )
    validator = MuJoCoCollisionValidator(
        mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE),
        joint_names=("arm_slide",),
        robot_root_body_name="robot_root",
        collision_margin_m=0.01,
        collision_model_version="attachment-v1",
        collision_contexts={"far": far, "near": near},
        entity_geoms={"part": ("part",)},
    )

    assert validator.check((0.0,), context=far).valid
    collision = validator.check((0.0,), context=near)

    assert not collision.valid
    assert collision.failure_code == "COLLISION_MARGIN_VIOLATION"


def test_gripper_joint_snapshot_changes_collision_geometry() -> None:
    open_context = CollisionContext(
        context_id="open",
        collision_model_version="attachment-v1",
        kinematic_joint_positions={"finger_slide": 0.0},
    )
    closed_context = CollisionContext(
        context_id="closed",
        collision_model_version="attachment-v1",
        kinematic_joint_positions={"finger_slide": 0.2},
    )
    validator = MuJoCoCollisionValidator(
        mujoco.MjModel.from_xml_string(_ATTACHMENT_SCENE),
        joint_names=("arm_slide",),
        robot_root_body_name="robot_root",
        collision_margin_m=0.01,
        collision_model_version="attachment-v1",
        collision_contexts={
            open_context.context_id: open_context,
            closed_context.context_id: closed_context,
        },
        entity_geoms={"part": ("part",)},
        allowed_collision_pairs=(("robot", "part"),),
    )

    assert validator.check((0.0,), context=open_context).valid
    closed = validator.check((0.0,), context=closed_context)

    assert not closed.valid
    assert any(
        {contact.geom_a, contact.geom_b} == {"finger_col", "finger_wall_col"}
        for contact in closed.contacts
        if not contact.allowed
    )


def test_robosuite_rack_scene_adapter_smoke() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("robosuite")
    from experiment.ee_rack_layout_demo import EERackLayoutEnv

    env = EERackLayoutEnv(has_renderer=False, has_offscreen_renderer=False)
    try:
        env.reset()
        entity_bodies = {
            ee: env.sim.model.body_id2name(body_id)
            for ee, body_id in env.ee_body_ids.items()
        }
        validator = MuJoCoCollisionValidator.from_robosuite_env(
            env,
            collision_margin_m=0.005,
            entity_body_names=entity_bodies,
        )
        initial_q = tuple(float(value) for value in env.sim.data._data.qpos[:6])

        result = validator.check(initial_q)
        rack_collision = validator.check(
            (5.3708, 0.2399, 0.7827, 0.1397, -5.4886, 0.7964)
        )

        assert result.valid
        assert len(validator.joint_names) == 6
        assert "robot0_forearm_col" in validator.robot_geom_names
        assert "gripper0_right_qc_plate" in validator.robot_geom_names
        assert not rack_collision.valid
        assert any(
            "rack" in contact.geom_a or "rack" in contact.geom_b
            for contact in rack_collision.contacts
        )
    finally:
        env.close()
