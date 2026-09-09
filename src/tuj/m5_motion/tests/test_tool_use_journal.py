"""Tool-Use-Journal compatibility without depending on an external checkout."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from tuj.m5_motion.tool_use_journal import (
    ToolUseJournalCollisionModelCompiler,
    ToolUseJournalCompatibilityError,
    ToolUseJournalEnvironmentAdapter,
    make_tool_use_journal_env,
    registered_tool_use_journal_environments,
)
from tuj.m5_motion.oracle import _as_position
from tuj.m5_motion.schema import (
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
from tuj.m5_motion.tool_use_journal_runtime import (
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
from tuj.m5_motion.runtime_checkpoint import (
    RuntimeCheckpointError,
    restore_runtime_checkpoint,
    save_runtime_checkpoint,
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


def test_repository_environment_registry_is_discovered_from_package_import(
    monkeypatch,
) -> None:
    import robosuite
    from robosuite.environments.base import REGISTERED_ENVS

    repository = Path(__file__).resolve().parents[4]
    registered = registered_tool_use_journal_environments(repository)
    assert {"C1_1_LegoSweep", "C2_1_ObjectSorting"} <= registered
    assert "Lift" not in registered

    experimental_type = type("ExperimentalTask", (), {})
    experimental_type.__module__ = "environments.experimental_task"
    monkeypatch.setitem(REGISTERED_ENVS, "ExperimentalTask", experimental_type)
    observed: dict[str, object] = {}

    def fake_make(*, env_name: str, **options):
        observed["env_name"] = env_name
        observed["options"] = options
        return SimpleNamespace(environment_name=env_name)

    monkeypatch.setattr(robosuite, "make", fake_make)

    env = make_tool_use_journal_env(
        repository,
        "ExperimentalTask",
        active_ee=None,
        seed=7,
    )

    assert "ExperimentalTask" in registered_tool_use_journal_environments(
        repository
    )
    assert env.environment_name == "ExperimentalTask"
    assert observed["env_name"] == "ExperimentalTask"
    assert observed["options"]["gripper_types"] is None
    assert observed["options"]["seed"] == 7


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


def test_finger_friction_enables_declared_torsional_and_rolling_terms() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand">
              <geom name="gripper0_right_left_fingertip_collision"
                    type="box" size="0.01 0.01 0.01"/>
              <geom name="gripper0_right_right_fingerpad_collision"
                    type="box" size="0.01 0.01 0.01" pos="0.03 0 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    runtime = object.__new__(ToolUseJournalEERuntime)
    runtime._active_ee = "2F"
    runtime._closed = False
    runtime._env = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=SimpleNamespace(_data=data),
        )
    )

    names = runtime.set_finger_gripper_contact_friction((1.0, 0.005, 0.0001))

    assert len(names) == 2
    for name in names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_friction[geom_id] == pytest.approx(
            (1.0, 0.005, 0.0001)
        )
        assert int(model.geom_condim[geom_id]) == 6


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


def test_compiler_reuses_variants_with_current_reference_state() -> None:
    variants = {
        active_ee: _fake_env(active_ee)
        for active_ee in (None, "2F", "3F", "vac")
    }
    compiler = ToolUseJournalCollisionModelCompiler.from_environments(
        variants["2F"], variants, source_revision="fixture"
    )
    live = _fake_env("2F")
    live.sim.data._data.qpos[0] = 0.43

    refreshed = compiler.with_reference_environment(live)
    compiled = refreshed.compile("2F")

    arm_joint_id = mujoco.mj_name2id(
        compiled.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]
    )
    qpos_index = int(compiled.model.jnt_qposadr[arm_joint_id])
    assert compiled.baseline_qpos[qpos_index] == pytest.approx(0.43)
    assert refreshed.attached_model_versions == compiler.attached_model_versions


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


def test_function_finger_hold_copies_targets_and_clears_on_ee_change() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    gripper = runtime.env.robots[0].gripper["right"]
    gripper.current_action = np.array([.37])
    runtime.command_gripper(engaged=True, suction=False)
    runtime.capture_gripper_hold()
    gripper.current_action[:] = .9
    assert runtime._captured_gripper_action == pytest.approx([.37])
    runtime.unlock("2F")
    assert runtime._captured_gripper_action is None
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

    assert (
        runtime.command_gripper(
            engaged=True,
            suction=False,
            command=0.0,
        )
        == 0.0
    )
    assert runtime.grasp_engaged is True
    assert runtime.command_gripper(engaged=True, suction=False) == 1.0
    runtime.hold_gripper_position()
    assert runtime.gripper_command == 1.0
    assert runtime.command_gripper(engaged=True, suction=False) == 1.0
    attachment = runtime.attach_object(
        "apple",
        max_attach_distance_m=0.05,
        max_attach_penetration_m=0.05,
    )
    assert attachment.object_id == "apple"
    assert runtime.attached_object_id == "apple"
    runtime.mark_attached_object_as_tool("apple")
    assert runtime.held_tool_id == "apple"
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
    assert runtime.held_tool_id is None
    runtime.close()


def test_runtime_checkpoint_restores_exact_attachment_without_new_grasp(
    tmp_path: Path,
) -> None:
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
    runtime.env.robots[0].gripper["right"].current_action = np.array([0.37])
    runtime.command_gripper(engaged=True, suction=False)
    runtime.capture_gripper_hold()
    runtime.attach_object(
        "apple",
        max_attach_distance_m=0.05,
        max_attach_penetration_m=0.05,
    )
    runtime.mark_attached_object_as_tool("apple")
    shoulder = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]
    )
    data.qpos[int(model.jnt_qposadr[shoulder])] = 0.3
    mujoco.mj_forward(model, data)
    runtime.synchronize_attached_object()
    expected_qpos = data.qpos.copy()
    checkpoint = tmp_path / "held-apple.checkpoint.json"
    save_runtime_checkpoint(
        checkpoint,
        runtime,
        progress={"completed_subgoals": ["acquire", "transport"]},
    )
    runtime.close()

    restored = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    result = restore_runtime_checkpoint(checkpoint, restored)

    restored_data = restored.env.sim.data._data
    assert restored_data.qpos == pytest.approx(expected_qpos)
    assert restored.attached_object_id == "apple"
    assert restored.held_tool_id == "apple"
    assert restored.captured_gripper_action == pytest.approx((0.37,))
    assert result.progress["completed_subgoals"] == ["acquire", "transport"]
    restored.close()


def test_runtime_checkpoint_rejects_wrong_active_ee(tmp_path: Path) -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    checkpoint = tmp_path / "two-finger.checkpoint.json"
    save_runtime_checkpoint(checkpoint, runtime)
    runtime.close()

    wrong = ToolUseJournalEERuntime(_fake_env("3F"), _fake_env)
    with pytest.raises(RuntimeCheckpointError, match="requires active EE"):
        restore_runtime_checkpoint(checkpoint, wrong)
    wrong.close()


def test_runtime_checkpoint_accepts_legacy_v1_after_named_layout_check(
    tmp_path: Path,
) -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    checkpoint = tmp_path / "legacy.checkpoint.json"
    save_runtime_checkpoint(checkpoint, runtime, progress={"step": 3})
    runtime.close()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload.pop("model_signature_kind")
    payload["model_signature"] = "unstable-full-mjcf-hash"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    restored = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    result = restore_runtime_checkpoint(checkpoint, restored)

    assert result.progress == {"step": 3}
    restored.close()


def test_runtime_checkpoint_rolls_back_physical_and_logical_state_on_failure(
    tmp_path: Path,
) -> None:
    source = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    model = source.env.sim.model._model
    data = source.env.sim.data._data
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
    source.command_gripper(engaged=True, suction=False)
    source.attach_object(
        "apple",
        max_attach_distance_m=0.05,
        max_attach_penetration_m=0.05,
    )
    checkpoint = tmp_path / "invalid-attachment.checkpoint.json"
    save_runtime_checkpoint(checkpoint, source)
    source.close()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["logical_state"]["attachment"]["position_in_reference_m"] = [
        1.0,
        1.0,
        1.0,
    ]
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    restored = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    restored_model = restored.env.sim.model._model
    restored_data = restored.env.sim.data._data
    shoulder = mujoco.mj_name2id(
        restored_model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS[0]
    )
    restored_data.qpos[int(restored_model.jnt_qposadr[shoulder])] = 0.6
    mujoco.mj_forward(restored_model, restored_data)
    before_qpos = restored_data.qpos.copy()

    with pytest.raises(RuntimeCheckpointError, match="rolled back"):
        restore_runtime_checkpoint(checkpoint, restored)

    assert restored_data.qpos == pytest.approx(before_qpos)
    assert restored.attached_object_id is None
    assert restored.held_tool_id is None
    assert restored.grasp_engaged is False
    restored.close()


def test_runtime_checkpoint_rejects_contact_hold_without_bilateral_contact(
    tmp_path: Path,
) -> None:
    source = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    source.command_gripper(engaged=True, suction=False)
    source.mark_contact_friction_object_as_tool("apple")
    checkpoint = tmp_path / "unsupported-contact-hold.checkpoint.json"
    save_runtime_checkpoint(checkpoint, source)
    source.close()

    restored = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    before_qpos = restored.env.sim.data._data.qpos.copy()

    with pytest.raises(RuntimeCheckpointError, match="bilateral finger contact"):
        restore_runtime_checkpoint(checkpoint, restored)

    assert restored.env.sim.data._data.qpos == pytest.approx(before_qpos)
    assert restored.held_tool_id is None
    assert restored.grasp_engaged is False
    restored.close()


def test_runtime_contact_friction_hold_has_no_synthetic_attachment() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)

    runtime.command_gripper(engaged=True, suction=False)
    runtime.mark_contact_friction_object_as_tool("apple")

    assert runtime.held_tool_id == "apple"
    assert runtime.attached_object_id is None
    with pytest.raises(ToolUseJournalRuntimeError, match="cannot attach"):
        runtime.attach_object("apple")
    with pytest.raises(ToolUseJournalRuntimeError, match="contact friction"):
        runtime.unlock("2F")

    runtime.command_gripper(engaged=False, suction=False)
    assert runtime.held_tool_id is None
    runtime.close()


def test_controller_config_uses_absolute_joint_targets() -> None:
    config = tool_use_journal_joint_position_controller_config(
        kp=80.0, damping_ratio=1.2
    )
    arm = config["body_parts"]["right"]

    assert arm["type"] == "JOINT_POSITION"
    assert arm["input_type"] == "absolute"
    assert arm["kp"] == pytest.approx(80.0)
    assert arm["damping_ratio"] == pytest.approx(1.2)
    assert arm["gripper"]["type"] == "GRIP"
    assert ToolUseJournalControllerTrajectoryPlayer._CONTROLLER_TRACKING is True
    with pytest.raises(ValueError, match="kp"):
        tool_use_journal_joint_position_controller_config(kp=0.0)


def test_controller_collision_check_stride_must_be_positive() -> None:
    runtime = SimpleNamespace()

    player = ToolUseJournalControllerTrajectoryPlayer(
        runtime, collision_check_stride=5
    )
    assert player._collision_check_stride == 5
    with pytest.raises(ValueError, match="positive integer"):
        ToolUseJournalControllerTrajectoryPlayer(
            runtime, collision_check_stride=0
        )


def test_controller_endpoint_event_group_waits_for_settle() -> None:
    run = _ee_exchange_simulation_run()
    segment = run.plan.segments[0]
    segment.metadata["motion_end_time_s"] = segment.end_time_s
    segment.metadata["tracking_settle"] = {
        "eef_tolerance_m": 0.005,
        "eef_orientation_tolerance_rad": 0.05,
    }
    events = [
        TrajectoryEvent(
            event_id="diagnostic-first",
            time_from_start_s=segment.end_time_s,
            event_type=EventType.WAIT,
        ),
        TrajectoryEvent(
            event_id="grip-second",
            time_from_start_s=segment.end_time_s,
            event_type=EventType.GRIPPER_CLOSE,
        ),
    ]

    assert ToolUseJournalControllerTrajectoryPlayer._event_group_waits_for_settle(
        run.plan, events, 0, {}
    )
    assert not ToolUseJournalControllerTrajectoryPlayer._event_group_waits_for_settle(
        run.plan,
        events,
        0,
        {segment.segment_id: {"settled": True}},
    )


def test_controller_does_not_step_past_unsettled_endpoint_event() -> None:
    env = _fake_env("2F")
    env.control_timestep = 0.02
    env.model_timestep = 0.002
    runtime = ToolUseJournalEERuntime(env, _fake_env)
    state = ToolUseJournalEnvironmentAdapter(env).world_snapshot().robot_state
    q = list(state.joint_positions_rad)
    boundary_s = 0.031
    duration_s = 0.051

    class _DeterministicBoundaryPlayer(ToolUseJournalControllerTrajectoryPlayer):
        def _controller_action(self, plan, desired_joint_positions):
            del plan, desired_joint_positions
            return np.zeros(1, dtype=float)

        def _advance_controller(self, action):
            del action
            data = self.runtime.env.sim.data._data
            data.time += self.runtime.env.control_timestep
            return float(data.time)

        def _execute_event(self, event):
            return f"executed {event.event_id}"

    plan = MotionPlan(
        plan_id="non-grid-settle-plan",
        request_id="non-grid-settle-request",
        provenance=_provenance(
            "non-grid-settle-plan-artifact",
            "MotionPlan",
            ModuleName.MOTION_PLANNER,
        ),
        scene_signature="non-grid-settle-scene",
        robot_id=state.robot_id,
        joint_names=list(state.joint_names),
        duration_s=duration_s,
        segments=[
            TrajectorySegment(
                segment_id="non-grid-grasp",
                segment_type=SegmentType.GRASP,
                start_time_s=0.0,
                end_time_s=boundary_s,
                collision_checked=False,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        joint_positions_rad=q,
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=boundary_s,
                        joint_positions_rad=q,
                    ),
                ],
                metadata={
                    "motion_end_time_s": boundary_s,
                    "tracking_settle": {
                        "joint_tolerance_rad": 0.001,
                        "max_wait_s": 0.1,
                        "required_consecutive_ticks": 3,
                    },
                },
            ),
            TrajectorySegment(
                segment_id="non-grid-retreat",
                segment_type=SegmentType.RETREAT,
                start_time_s=boundary_s,
                end_time_s=duration_s,
                collision_checked=False,
                waypoints=[
                    TrajectoryWaypoint(
                        time_from_start_s=boundary_s,
                        joint_positions_rad=q,
                    ),
                    TrajectoryWaypoint(
                        time_from_start_s=duration_s,
                        joint_positions_rad=q,
                    ),
                ],
            ),
        ],
        events=[
            TrajectoryEvent(
                event_id="attach-at-non-grid-endpoint",
                time_from_start_s=boundary_s,
                event_type=EventType.ATTACH_OBJECT,
                target_id="apple",
            )
        ],
        expected_final_state=state.model_copy(
            update={"joint_velocities_rad_s": [0.0] * len(q)}
        ),
    )
    run = SimulationRun(
        run_id="non-grid-settle-run",
        provenance=_provenance(
            "non-grid-settle-run-artifact",
            "SimulationRun",
            ModuleName.SIMULATOR,
        ),
        plan=plan,
        config=SimulationConfig(
            physics_timestep_s=env.model_timestep,
            control_timestep_s=env.control_timestep,
            max_duration_s=0.2,
        ),
    )

    try:
        report = _DeterministicBoundaryPlayer(runtime).execute(run)
    finally:
        runtime.close()

    assert report.status is ExecutionStatus.SUCCESS
    assert [event.event_id for event in report.executed_events] == [
        "attach-at-non-grid-endpoint"
    ]
    assert report.metadata["segment_tracking"][0]["segment_id"] == (
        "non-grid-grasp"
    )
    assert report.metadata["segment_tracking"][0][
        "adaptive_settle_succeeded"
    ] is True


def test_controller_settle_validates_orientation_tolerance() -> None:
    run = _ee_exchange_simulation_run()
    segment = run.plan.segments[0]
    segment.metadata["tracking_settle"] = {
        "eef_orientation_tolerance_rad": 0.05,
    }

    assert ToolUseJournalControllerTrajectoryPlayer._tracking_settle_config(
        segment
    ) == {
        "eef_orientation_tolerance_rad": 0.05,
        "max_wait_s": 2.0,
        "required_consecutive_ticks": 3,
    }
    assert ToolUseJournalControllerTrajectoryPlayer._eef_orientation_error_rad(
        (0.0, 0.0, 0.0, 1.0), np.eye(3)
    ) == pytest.approx(0.0)
    angle = 0.1
    actual = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert ToolUseJournalControllerTrajectoryPlayer._eef_orientation_error_rad(
        (0.0, 0.0, 0.0, 1.0), actual
    ) == pytest.approx(angle)


def test_live_joint_controller_gains_can_be_retuned_between_phases() -> None:
    runtime = ToolUseJournalEERuntime(_fake_env("2F"), _fake_env)
    controller = SimpleNamespace(name="JOINT_POSITION", control_dim=6)
    runtime.env.robots[0].part_controllers = {"right": controller}

    runtime.set_joint_position_controller_gains(kp=80.0, damping_ratio=1.0)

    assert controller.kp == pytest.approx([80.0] * 6)
    assert controller.kd == pytest.approx([2.0 * np.sqrt(80.0)] * 6)
    runtime.close()


def test_controller_player_advances_the_real_robosuite_physics_loop() -> None:
    pytest.importorskip("robosuite")
    repository = Path(__file__).resolve().parents[4]
    if not repository.is_dir():
        pytest.skip("workspace Tool-Use-Journal checkout is unavailable")
    runtime = ToolUseJournalEERuntime.from_repository_for_controller(
        repository,
        "C1_1_LegoSweep",
        active_ee="2F",
        seed=0,
        ignore_done=True,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
    )
    try:
        state = ToolUseJournalEnvironmentAdapter(runtime.env).world_snapshot().robot_state
        q = list(state.joint_positions_rad)
        duration = 2.0 * float(runtime.env.control_timestep)
        plan = MotionPlan(
            plan_id="controller-physics-smoke-plan",
            request_id="controller-physics-smoke-request",
            provenance=_provenance(
                "controller-physics-plan-artifact",
                "MotionPlan",
                ModuleName.MOTION_PLANNER,
            ),
            scene_signature="controller-physics-smoke-scene",
            robot_id=state.robot_id,
            joint_names=list(state.joint_names),
            duration_s=duration,
            segments=[
                TrajectorySegment(
                    segment_id="controller-physics-stationary",
                    segment_type=SegmentType.CUSTOM,
                    start_time_s=0.0,
                    end_time_s=duration,
                    collision_checked=False,
                    waypoints=[
                        TrajectoryWaypoint(
                            time_from_start_s=0.0,
                            joint_positions_rad=q,
                        ),
                        TrajectoryWaypoint(
                            time_from_start_s=duration,
                            joint_positions_rad=q,
                        ),
                    ],
                )
            ],
            expected_final_state=state.model_copy(
                update={"joint_velocities_rad_s": [0.0] * len(q)}
            ),
        )
        run = SimulationRun(
            run_id="controller-physics-smoke-run",
            provenance=_provenance(
                "controller-physics-run-artifact",
                "SimulationRun",
                ModuleName.SIMULATOR,
            ),
            plan=plan,
            config=SimulationConfig(
                physics_timestep_s=float(runtime.env.model_timestep),
                control_timestep_s=float(runtime.env.control_timestep),
                max_duration_s=duration + 1.0,
            ),
        )

        report = ToolUseJournalControllerTrajectoryPlayer(runtime).execute(run)

        assert report.status is ExecutionStatus.SUCCESS
        assert report.failure is None
        assert report.metrics.executed_duration_s >= duration
        assert report.metadata["controller_tracking_simulated"] is True
        assert report.metadata["playback_mode"] == (
            "ROBOSUITE_ABSOLUTE_JOINT_POSITION_CONTROLLER"
        )
    finally:
        runtime.close()


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


def test_metric_preshape_quantizes_to_adjacent_wider_2f_state() -> None:
    robot = SimpleNamespace(
        action_dim=2,
        composite_controller=SimpleNamespace(
            _action_split_indexes={
                "right": (0, 1),
                "right_gripper": (1, 2),
            }
        ),
        robot_model=SimpleNamespace(joints=("joint",)),
    )
    aperture_states = (0.04, 0.022)
    state_index = 0
    commands: list[dict[str, object]] = []
    runtime = SimpleNamespace(
        active_ee="2F",
        env=SimpleNamespace(robots=[robot]),
        fingerpad_separation_m=lambda: aperture_states[state_index],
        command_gripper=lambda **kwargs: commands.append(kwargs),
    )
    player = object.__new__(ToolUseJournalControllerTrajectoryPlayer)
    player.runtime = runtime
    player._actual_joint_positions = lambda env, names: np.zeros(len(names))

    def advance(action: np.ndarray) -> float:
        nonlocal state_index
        if action[1] > 0.0:
            state_index = min(state_index + 1, len(aperture_states) - 1)
        elif action[1] < 0.0:
            state_index = max(state_index - 1, 0)
        return 0.0

    player._advance_controller = advance

    actual = player.preshape_finger_gripper_to_aperture(
        target_aperture_m=0.031,
        tolerance_m=0.002,
        settle_ticks_per_iteration=1,
        final_settle_ticks=1,
        allow_wider_discrete_state=True,
    )

    assert actual == pytest.approx(0.04)
    assert actual > 0.031
    assert commands == [
        {"engaged": True, "suction": False, "command": 0.0}
    ]


def test_metric_preshape_can_reject_discrete_quantization() -> None:
    robot = SimpleNamespace(
        action_dim=2,
        composite_controller=SimpleNamespace(
            _action_split_indexes={
                "right": (0, 1),
                "right_gripper": (1, 2),
            }
        ),
        robot_model=SimpleNamespace(joints=("joint",)),
    )
    aperture_states = (0.04, 0.022)
    state_index = 0
    runtime = SimpleNamespace(
        active_ee="2F",
        env=SimpleNamespace(robots=[robot]),
        fingerpad_separation_m=lambda: aperture_states[state_index],
        command_gripper=lambda **kwargs: None,
    )
    player = object.__new__(ToolUseJournalControllerTrajectoryPlayer)
    player.runtime = runtime
    player._actual_joint_positions = lambda env, names: np.zeros(len(names))

    def advance(action: np.ndarray) -> float:
        nonlocal state_index
        if action[1] > 0.0:
            state_index = min(state_index + 1, len(aperture_states) - 1)
        elif action[1] < 0.0:
            state_index = max(state_index - 1, 0)
        return 0.0

    player._advance_controller = advance

    with pytest.raises(
        ToolUseJournalRuntimeError,
        match="between discrete 2F controller states",
    ):
        player.preshape_finger_gripper_to_aperture(
            target_aperture_m=0.031,
            tolerance_m=0.002,
            settle_ticks_per_iteration=1,
            final_settle_ticks=1,
            allow_wider_discrete_state=False,
        )


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
