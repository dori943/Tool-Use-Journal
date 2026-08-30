"""Tool-Use-Journal compatibility without depending on an external checkout."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tuj.m4_motion.tool_use_journal import (
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
)
from tuj.m4_motion.oracle import _as_position
from tuj.m4_motion.schema import (
    ArtifactProvenance,
    CollisionContext,
    EventExecutionStatus,
    EventType,
    ExecutionStatus,
    ModuleName,
    MotionPlan,
    RobotState,
    SegmentType,
    SimulationConfig,
    SimulationRun,
    TrajectoryEvent,
    TrajectorySegment,
    TrajectoryWaypoint,
)
from tuj.m4_motion.tool_use_journal_runtime import (
    AttachmentContactMetrics,
    AttachmentMode,
    BreakableWeldConfig,
    ToolUseJournalAttachmentBroken,
    ToolUseJournalControllerTrajectoryPlayer,
    ToolUseJournalEERuntime,
    ToolUseJournalKinematicTrajectoryPlayer,
    ToolUseJournalRuntimeError,
    tool_use_journal_joint_position_controller_config,
)


ARM_JOINTS = (
    "robot0_shoulder_pan_joint",
    "robot0_shoulder_lift_joint",
    "robot0_elbow_joint",
    "robot0_wrist_1_joint",
    "robot0_wrist_2_joint",
    "robot0_wrist_3_joint",
)

EE_ROOTS = {
    "3F": "gripperrack_3F_right_gripper",
    "vac": "gripperrack_vac_vacuum_base",
    "2F": "gripperrack_2F_robotiq_85_adapter_link",
}

MOUNTED_ROOTS = {
    None: "gripper0_right_null_gripper",
    "2F": "gripper0_right_robotiq_85_adapter_link",
    "3F": "gripper0_right_right_gripper",
    "vac": "gripper0_right_vacuum_base",
}

GRIPPER_CLASS_NAMES = {
    None: "NullGripper",
    "2F": "Robotiq85Gripper",
    "3F": "JacoThreeFingerDexterousGripper",
    "vac": "VacuumGripper",
}


def _arm_xml(mounted_root: str, active_ee: str | None) -> str:
    opened = []
    closed = []
    for index, joint_name in enumerate(ARM_JOINTS):
        opened.append(
            f'<body name="robot0_link_{index}" pos="0 0 0.08">'
            f'<joint name="{joint_name}" type="hinge" axis="0 0 1" '
            'range="-6.28 6.28"/>'
            f'<geom name="robot0_link_{index}_collision" type="capsule" '
            'size="0.018 0.035" pos="0 0 0.04" contype="1" conaffinity="1"/>'
        )
        closed.append("</body>")
    mounted_geom = (
        f'<geom name="mounted_{active_ee}_collision" type="box" '
        'size="0.02 0.02 0.03" pos="0 0 0.03" '
        'contype="1" conaffinity="1"/>'
        if active_ee is not None
        else ""
    )
    opened.append(
        '<body name="robot0_right_hand" pos="0 0 0.08">'
        '<geom name="robot0_hand_collision" type="sphere" size="0.02" '
        'contype="1" conaffinity="1"/>'
        f'<body name="{mounted_root}">{mounted_geom}</body>'
        '</body>'
    )
    return "".join(opened + list(reversed(closed)))


def _source_xml(active_ee: str | None) -> str:
    rack_ees = "".join(
        f'<body name="{root}" pos="{0.3 + index * 0.1} -0.5 0.8">'
        f'<geom name="gripperrack_{ee}_collision" type="box" '
        'size="0.025 0.025 0.05" group="0" contype="0" conaffinity="0"/>'
        '</body>'
        for index, (ee, root) in enumerate(EE_ROOTS.items())
    )
    supports = "".join(
        f'<body name="ee_rack_slot_{ee}" pos="{index * 0.1} 0 0">'
        f'<geom name="ee_rack_support_{ee}" type="box" '
        'size="0.03 0.03 0.04" group="1" contype="0" conaffinity="0"/>'
        '</body>'
        for index, ee in enumerate(("3F", "vac", "2F"))
    )
    return f"""
<mujoco model="tool_use_journal_fixture">
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="table_collision" type="box" size="1 1 0.04"
          pos="0 0 0.4" contype="1" conaffinity="1"/>
    <body name="robot0_base" pos="0 0 0.5">
      {_arm_xml(MOUNTED_ROOTS[active_ee], active_ee)}
    </body>
    {rack_ees}
    <body name="ee_rack" pos="0.4 -0.5 0.4">
      <geom name="ee_rack_base" type="box" size="0.3 0.1 0.02"
            group="1" contype="0" conaffinity="0"/>
      {supports}
    </body>
    <body name="apple" pos="0.2 0.1 0.5">
      <freejoint name="apple_joint"/>
      <geom name="apple_collision" type="sphere" size="0.03"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _ModelWrapper:
    def __init__(self, xml: str) -> None:
        self._xml = xml

    def get_xml(self) -> str:
        return self._xml


def _fake_env(active_ee: str | None):
    xml = _source_xml(active_ee)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper_type = type(GRIPPER_CLASS_NAMES[active_ee], (), {})
    gripper = gripper_type()
    gripper.root_body = MOUNTED_ROOTS[active_ee]
    robot_model = SimpleNamespace(joints=ARM_JOINTS, root_body="robot0_base")
    robot = SimpleNamespace(robot_model=robot_model, gripper={"right": gripper})

    class C2_1_ObjectSorting:
        def reset(self):
            mujoco.mj_forward(model, data)
            return {}

        def close(self):
            return None

    env = C2_1_ObjectSorting()
    env.sim = SimpleNamespace(
        model=SimpleNamespace(_model=model),
        data=SimpleNamespace(_data=data),
    )
    env.model = _ModelWrapper(xml)
    env.robots = [robot]
    env.robot_spec = {"robot_id": "ur5e_0", "current_ee": "2F"}
    env.current_ee_id = "2F"
    env.obj_body_id = {
        "apple": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    }
    env.ee_rack_info = {
        ee: {
            "rack_body": root,
            "rack_slot": f"ee_rack_slot_{ee}",
            "rack_position": np.asarray((0.3 + index * 0.1, -0.5, 0.8)),
            "rack_orientation": np.asarray((0.0, 1.0, 0.0, 0.0)),
            "rack_support_top_z": 0.5,
            "rack_support_height": 0.08,
            "gripper_class": GRIPPER_CLASS_NAMES[ee],
        }
        for index, (ee, root) in enumerate(EE_ROOTS.items())
    }
    return env


def test_adapter_captures_mjcf_world_and_target_ee_ids() -> None:
    env = _fake_env("2F")
    adapter = ToolUseJournalEnvironmentAdapter(env, source_revision="fixture")

    world = adapter.world_snapshot()

    assert world.robot_state.robot_id == "ur5e_0"
    assert world.robot_state.joint_names == list(ARM_JOINTS)
    assert set(world.rack) == {"2F", "3F", "vac"}
    assert world.rack["2F"]["dock_pose"]["orientation_xyzw"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert world.objects["apple"]["dimensions_m"] == pytest.approx(
        [0.06, 0.06, 0.06]
    )
    assert world.metadata["physical_active_ee"] == "2F"
    assert world.metadata["ee_metadata_matches_physics"] is True
    assert adapter.make_kinematics().base_position_m == pytest.approx(
        (0.0, 0.0, 0.5)
    )
    assert _as_position(world.objects["apple"]) == pytest.approx(
        world.objects["apple"]["pose"]["position_m"]
    )
    assert _as_position(world.rack["2F"]) == pytest.approx(
        world.rack["2F"]["dock_pose"]["position_m"]
    )


def test_adapter_rejects_declared_ee_that_is_physically_bare() -> None:
    adapter = ToolUseJournalEnvironmentAdapter(_fake_env(None))

    assert adapter.ee_metadata_matches_physics is False
    with pytest.raises(ToolUseJournalCompatibilityError, match="does not match"):
        adapter.require_physical_ee()


def test_compiler_promotes_rack_and_removes_active_display_duplicate() -> None:
    variants = {
        active_ee: _fake_env(active_ee)
        for active_ee in (None, "2F", "3F", "vac")
    }
    compiler = ToolUseJournalCollisionModelCompiler.from_environments(
        variants["2F"], variants, source_revision="fixture"
    )

    bare = compiler.compile(None)
    attached = compiler.compile("2F")

    assert "ee_rack_base" in bare.promoted_rack_geom_names
    rack_geom = mujoco.mj_name2id(
        bare.model, mujoco.mjtObj.mjOBJ_GEOM, "ee_rack_support_2F"
    )
    assert int(bare.model.geom_contype[rack_geom]) == 1
    assert (
        mujoco.mj_name2id(
            attached.model,
            mujoco.mjtObj.mjOBJ_BODY,
            EE_ROOTS["2F"],
        )
        < 0
    )
    entities = dict(attached.entity_selectors)
    assert entities["2F"] == (MOUNTED_ROOTS["2F"],)
    assert entities["rack_support:2F"] == ("ee_rack_support_2F",)


def test_exchange_contexts_route_to_target_scene_models() -> None:
    variants = {
        active_ee: _fake_env(active_ee)
        for active_ee in (None, "2F", "3F", "vac")
    }
    compiler = ToolUseJournalCollisionModelCompiler.from_environments(
        variants["2F"], variants, source_revision="fixture"
    )
    contexts = compiler.build_ee_exchange_contexts(from_ee="2F", to_ee="vac")

    registry = compiler.build_collision_registry(
        contexts, collision_margin_m=0.0
    )

    assert registry.joint_names == ARM_JOINTS
    assert contexts["bare-flange"].collision_model_version == (
        compiler.bare_model_version
    )
    assert contexts["ee-attached:vac"].collision_model_version == (
        compiler.attached_model_versions["vac"]
    )


def _joint_qpos(env, joint_name: str) -> np.ndarray:
    model = env.sim.model._model
    data = env.sim.data._data
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    start = int(model.jnt_qposadr[joint_id])
    width = (
        7
        if int(model.jnt_type[joint_id])
        == int(mujoco.mjtJoint.mjJNT_FREE)
        else 1
    )
    return np.asarray(data.qpos[start : start + width], dtype=float).copy()


def test_runtime_swaps_physical_model_and_preserves_named_state() -> None:
    initial = _fake_env("2F")
    initial.sim.data._data.qpos[0] = 0.37
    apple_before = _joint_qpos(initial, "apple_joint")
    initial.sim.data._data.time = 1.25
    runtime = ToolUseJournalEERuntime(initial, _fake_env)

    unlock = runtime.unlock("2F")

    assert runtime.active_ee is None
    assert unlock.from_ee == "2F" and unlock.to_ee is None
    assert runtime.rack_ee_visible("2F") is True
    assert _joint_qpos(runtime.env, ARM_JOINTS[0]) == pytest.approx([0.37])
    assert _joint_qpos(runtime.env, "apple_joint") == pytest.approx(apple_before)
    assert runtime.env.sim.data._data.time == pytest.approx(1.25)

    lock = runtime.lock("vac")

    assert runtime.active_ee == "vac"
    assert lock.hidden_rack_ee == "vac"
    assert runtime.rack_ee_visible("vac") is False
    assert _joint_qpos(runtime.env, ARM_JOINTS[0]) == pytest.approx([0.37])
    assert _joint_qpos(runtime.env, "apple_joint") == pytest.approx(apple_before)
    runtime.verify_tool_lock("vac")
    runtime.close()


def test_runtime_model_swap_is_atomic_when_factory_builds_wrong_ee() -> None:
    initial = _fake_env("2F")

    def wrong_factory(_active_ee):
        return _fake_env("3F")

    runtime = ToolUseJournalEERuntime(initial, wrong_factory)

    with pytest.raises(ToolUseJournalRuntimeError, match="requested None"):
        runtime.unlock("2F")

    assert runtime.env is initial
    assert runtime.active_ee == "2F"
    runtime.close()


def test_runtime_grasp_attach_tracks_hand_and_blocks_tool_exchange() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    model = runtime.env.sim.model._model
    data = runtime.env.sim.data._data
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand"
    )
    apple_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "apple_joint"
    )
    apple_qpos = int(model.jnt_qposadr[apple_joint])
    data.qpos[apple_qpos : apple_qpos + 3] = data.xpos[hand_id]
    data.qpos[apple_qpos + 3 : apple_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    assert runtime.command_gripper(engaged=True, suction=False) == 1.0
    attachment = runtime.attach_object(
        "apple",
        max_attach_distance_m=0.05,
        max_attach_penetration_m=0.05,
    )
    assert attachment.object_id == "apple"
    assert runtime.attached_object_id == "apple"
    with pytest.raises(ToolUseJournalRuntimeError, match="while object"):
        runtime.unlock("2F")
    with pytest.raises(ToolUseJournalRuntimeError, match="detach object"):
        runtime.command_gripper(engaged=False, suction=False)

    before = data.qpos[apple_qpos : apple_qpos + 7].copy()
    shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]
    )
    data.qpos[int(model.jnt_qposadr[shoulder])] = 0.3
    mujoco.mj_forward(model, data)
    runtime.synchronize_attached_object()
    after = data.qpos[apple_qpos : apple_qpos + 7].copy()
    assert not np.allclose(after, before)

    runtime.detach_object("apple")
    runtime.command_gripper(engaged=False, suction=False)
    assert runtime.attached_object_id is None
    runtime.close()


def test_controller_config_uses_absolute_joint_targets() -> None:
    config = tool_use_journal_joint_position_controller_config(
        kp=75.0,
        damping_ratio=0.8,
    )
    arm = config["body_parts"]["right"]

    assert arm["type"] == "JOINT_POSITION"
    assert arm["input_type"] == "absolute"
    assert arm["kp"] == 75.0
    assert arm["damping_ratio"] == 0.8
    assert arm["gripper"]["type"] == "GRIP"
    assert ToolUseJournalControllerTrajectoryPlayer._CONTROLLER_TRACKING is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kp": 0.0}, "kp must be finite"),
        ({"damping_ratio": 0.0}, "damping_ratio must be finite"),
    ],
)
def test_controller_config_rejects_invalid_gains(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tool_use_journal_joint_position_controller_config(**kwargs)


def test_breakable_weld_can_require_opposed_finger_contacts() -> None:
    config = BreakableWeldConfig.from_parameters(
        {
            "require_retention_contact": False,
            "min_contact_count": 2,
            "min_normal_force_n": 1.0,
            "required_contact_groups": ["left_finger", "right_finger"],
        }
    )
    assert config.required_contact_groups == (
        "left_finger",
        "right_finger",
    )
    assert config.require_retention_contact is False
    valid = AttachmentContactMetrics(
        contact_count=4,
        normal_force_n=8.0,
        contact_groups=("left_finger", "right_finger"),
    )
    one_sided = AttachmentContactMetrics(
        contact_count=2,
        normal_force_n=8.0,
        contact_groups=("left_finger",),
    )

    assert not ToolUseJournalEERuntime._contact_contract_failures(
        valid, config
    )
    assert ToolUseJournalEERuntime._contact_contract_failures(
        one_sided, config
    ) == ("CONTACT_GROUPS:right_finger",)


def test_breakable_weld_detaches_when_required_torque_exceeds_limit() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    model = runtime.env.sim.model._model
    data = runtime.env.sim.data._data
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand"
    )
    apple_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "apple_joint"
    )
    apple_qpos = int(model.jnt_qposadr[apple_joint])
    data.qpos[apple_qpos : apple_qpos + 3] = data.xpos[hand_id]
    data.qpos[apple_qpos + 3 : apple_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    runtime.command_gripper(engaged=True, suction=False)
    runtime.attach_object(
        "apple",
        max_attach_distance_m=0.05,
        max_attach_penetration_m=0.05,
        attachment_mode=AttachmentMode.BREAKABLE_WELD,
        breakable_weld=BreakableWeldConfig(
            max_weld_torque_nm=1e-4,
            require_contact=False,
            startup_grace_steps=0,
            break_debounce_steps=1,
        ),
    )

    shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]
    )
    data.qpos[int(model.jnt_qposadr[shoulder])] = 0.3
    mujoco.mj_forward(model, data)
    with pytest.raises(
        ToolUseJournalAttachmentBroken, match="WELD_TORQUE_LIMIT"
    ):
        runtime.prepare_attachment_step()

    assert runtime.attached_object_id is None
    assert runtime.last_attachment_break is not None
    assert "WELD_TORQUE_LIMIT" in runtime.last_attachment_break.reasons
    runtime.close()


def _provenance(
    artifact_id: str,
    artifact_type: str,
    module: ModuleName,
) -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        produced_by=module,
        invocation_id=f"{artifact_id}:invocation",
    )


def _ee_exchange_simulation_run() -> SimulationRun:
    attached_2f = CollisionContext(
        context_id="ee-attached:2F",
        active_ee="2F",
        collision_model_version="fixture-2F",
    )
    bare = CollisionContext(
        context_id="bare-flange",
        collision_model_version="fixture-bare",
    )
    attached_vac = CollisionContext(
        context_id="ee-attached:vac",
        active_ee="vac",
        collision_model_version="fixture-vac",
    )
    first = TrajectorySegment(
        segment_id="undock",
        segment_type=SegmentType.EE_UNDOCK,
        start_time_s=0.0,
        end_time_s=1.0,
        collision_checked=True,
        collision_context_before=attached_2f,
        collision_context_after=bare,
        waypoints=[
            TrajectoryWaypoint(
                time_from_start_s=0.0,
                joint_positions_rad=[0.0] * len(ARM_JOINTS),
            ),
            TrajectoryWaypoint(
                time_from_start_s=1.0,
                joint_positions_rad=[0.1] * len(ARM_JOINTS),
            ),
        ],
    )
    second = TrajectorySegment(
        segment_id="dock",
        segment_type=SegmentType.EE_DOCK,
        start_time_s=1.0,
        end_time_s=2.0,
        collision_checked=True,
        collision_context_before=bare,
        collision_context_after=attached_vac,
        waypoints=[
            TrajectoryWaypoint(
                time_from_start_s=1.0,
                joint_positions_rad=[0.1] * len(ARM_JOINTS),
            ),
            TrajectoryWaypoint(
                time_from_start_s=2.0,
                joint_positions_rad=[0.2] * len(ARM_JOINTS),
            ),
        ],
    )
    plan = MotionPlan(
        plan_id="ee-exchange-plan",
        request_id="request",
        provenance=_provenance(
            "plan-artifact", "MotionPlan", ModuleName.MOTION_PLANNER
        ),
        scene_signature="fixture-scene",
        robot_id="ur5e_0",
        joint_names=list(ARM_JOINTS),
        duration_s=2.0,
        segments=[first, second],
        events=[
            TrajectoryEvent(
                event_id="unlock",
                time_from_start_s=1.0,
                event_type=EventType.TOOL_UNLOCK,
                target_id="2F",
            ),
            TrajectoryEvent(
                event_id="verify-release",
                time_from_start_s=1.0,
                event_type=EventType.VERIFY_TOOL_RELEASE,
                target_id="2F",
            ),
            TrajectoryEvent(
                event_id="lock",
                time_from_start_s=2.0,
                event_type=EventType.TOOL_LOCK,
                target_id="vac",
            ),
            TrajectoryEvent(
                event_id="verify-lock",
                time_from_start_s=2.0,
                event_type=EventType.VERIFY_TOOL_LOCK,
                target_id="vac",
            ),
        ],
        expected_final_state=RobotState(
            robot_id="ur5e_0",
            joint_names=list(ARM_JOINTS),
            joint_positions_rad=[0.2] * len(ARM_JOINTS),
            joint_velocities_rad_s=[0.0] * len(ARM_JOINTS),
        ),
    )
    return SimulationRun(
        run_id="runtime-run",
        provenance=_provenance(
            "run-artifact", "SimulationRun", ModuleName.SIMULATOR
        ),
        plan=plan,
        config=SimulationConfig(render=False),
    )


def test_player_replays_plan_and_executes_ee_exchange_events() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    player = ToolUseJournalKinematicTrajectoryPlayer(runtime)

    report = player.execute(_ee_exchange_simulation_run())

    assert report.status is ExecutionStatus.SUCCESS
    assert report.failure is None
    assert runtime.active_ee == "vac"
    assert len(runtime.transitions) == 2
    assert [event.status for event in report.executed_events] == [
        EventExecutionStatus.SUCCESS,
        EventExecutionStatus.SUCCESS,
        EventExecutionStatus.SUCCESS,
        EventExecutionStatus.SUCCESS,
    ]
    assert report.final_robot_state is not None
    assert report.final_robot_state.joint_positions_rad == pytest.approx(
        [0.2] * len(ARM_JOINTS)
    )
    assert report.metadata["controller_tracking_simulated"] is False
    runtime.close()
