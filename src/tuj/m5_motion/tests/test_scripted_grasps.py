from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tuj.m5_motion.schema import (
    ArtifactProvenance, AttachedObjectTransform, CollisionContext, MotionGoal,
    MotionPlanRequest, MotionTask, Pose, RobotState, SceneRef, WorldSnapshot,
)
from tuj.m5_motion.scripted_grasps.registry import (
    ENABLED_ENTRIES, ENTRIES, EXPERIMENTAL_INTEGRATION, PENDING_INTEGRATION,
    integration_status, resolve,
)
from tuj.m5_motion.scripted_grasps.frames import transform
from tuj.m5_motion.tool_use_journal_runtime import (
    ToolUseJournalEERuntime, ToolUseJournalKinematicTrajectoryPlayer, ToolUseJournalRuntimeError,
)
from tuj.m5_motion.tool_use_journal_planning import ToolUseJournalCollisionContextFactory


def request_for(entry=ENTRIES[0], action="PICK", object_id=None):
    target = object_id or entry.object_id
    return MotionPlanRequest(request_id="request",
        provenance=ArtifactProvenance(artifact_id="request", artifact_type="MotionPlanRequest",
            produced_by="TASK_PLANNER", invocation_id="test"),
        world=WorldSnapshot(scene=SceneRef(signature="before"),
            robot_state=RobotState(robot_id="robot", joint_names=["j1"], joint_positions_rad=[0.]),
            metadata={"environment_name": entry.environment, "physical_active_ee": entry.ee}),
        task=MotionTask(task_id="task", subgoal_id="pick", action_type=action, ee=entry.ee,
            target_ids=[target], goal=MotionGoal(goal_type="POSE", target_object_id=target),
            metadata={"attach_target": False}))


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e.object_id}-{e.environment}-{e.ee}")
def test_exact_scene_object_and_ee_dispatch(entry):
    from tuj.m5_motion.scripted_grasps.registry import ScriptedGraspUnavailable
    for request in (request_for(entry), request_for(entry, object_id=f"obj_{entry.object_id}_{entry.object_id}")):
        if entry.object_id in PENDING_INTEGRATION:
            with pytest.raises(ScriptedGraspUnavailable, match="NOT_VALIDATED"):
                resolve(request)
        else:
            assert resolve(request) == entry
    assert callable(entry.function())
    assert entry.recipe().ee_id == entry.ee
    request = request_for(entry, action="MOVE")
    assert resolve(request) is None
    request = request_for(entry)
    request.world.metadata["environment_name"] = "different-scene"
    assert resolve(request) is None


def test_excluded_unknown_and_wrong_hand():
    spoon_routes={(e.environment,e.ee) for e in ENTRIES if e.object_id=='spoon'}
    assert spoon_routes >= {
        ('C1_2_DoughFlatten','2F'),('C1_2_DoughFlatten','3F'),
        ('C2_1_ObjectSorting','2F'),('C2_1_ObjectSorting','3F'),
        ('C3_1_ObjectSorting','2F'),('C3_1_ObjectSorting','3F'),
        ('C3_2_BreakfastTrayPreparation','3F'),
    }
    assert not {"tongs", "ladle"} & {e.object_id for e in ENTRIES}
    assert resolve(request_for(object_id="plate_large")) is None
    assert resolve(request_for(object_id="tongs")) is None
    request = request_for()
    request.task.ee = "3F"
    # EE mismatch falls back to generic M5 grasp routing (origin/main).
    assert resolve(request) is None


def _c3_2_entry(object_id):
    return next(
        e for e in ENTRIES
        if e.object_id == object_id and e.environment == "C3_2_BreakfastTrayPreparation"
    )


def test_c3_2_plate_vac_and_fork_resolve_multi_instance():
    plate = _c3_2_entry("plate")
    fork = _c3_2_entry("fork")
    assert plate.ee == "vac" and plate.module_name == "plate_vac"
    assert fork.ee == "3F"
    for instance in ("plate_a", "plate_b"):
        resolved = resolve(request_for(plate, object_id=instance))
        assert resolved.object_id == "plate"
        assert resolved.module_name == "plate_vac"
        assert resolved.ee == "vac"
        assert resolved.scene_object_id == instance
        assert resolved.body_object_id == instance
        assert resolved.recipe().recipe_id.startswith("plate_vac_")
    for instance in ("fork_a", "fork_b"):
        resolved = resolve(request_for(fork, object_id=instance))
        assert resolved.object_id == "fork"
        assert resolved.ee == "3F"
        assert resolved.scene_object_id == instance
        assert resolved.body_object_id == instance
        assert resolved.recipe().object_id == "fork"
    # Exact type id still resolves without rewriting body id.
    assert resolve(request_for(plate)).body_object_id is None
    assert resolve(request_for(fork)).body_object_id is None
    # C1_1 2F plate remains unchanged.
    c1_plate = ENTRIES[0]
    assert c1_plate.environment == "C1_1_LegoSweep" and c1_plate.ee == "2F"
    assert resolve(request_for(c1_plate)) == c1_plate
    assert c1_plate.recipe().recipe_id == "plate_2f_calibrated_center_v5"
    # Explicit C1_1 vac ENTRIES row still resolves when M4 selects vac.
    c1_vac = next(
        entry
        for entry in ENTRIES
        if entry.object_id == "plate"
        and entry.environment == "C1_1_LegoSweep"
        and entry.ee == "vac"
    )
    assert resolve(request_for(c1_vac)) == c1_vac
    assert c1_vac.module_name == "plate_vac"
    assert c1_vac.recipe_name == "plate_vac_c2_1_recipe"
    assert c1_vac.recipe().task_id == "c2_1"
    # grasp_plate_vac must reuse the bound recipe (not default C3_2).
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.objects import plate_vac

    bound = c1_vac.recipe()
    calls = []

    class _Ctx:
        object_id = "plate"
        recipe = bound

        def execute_object(self, recipe):
            calls.append(recipe)
            return {"status": "SUCCESS"}

    plate_vac.grasp_plate_vac(_Ctx())
    assert calls == [bound]


def test_c3_2_remaining_instances_resolve_and_build_targets():
    """bread/fruit/spoon/mug *_a/*_b → type recipe + PRE/GRASP/LIFT."""
    from tuj.m5_motion.scripted_grasps.catalog_types import build_catalog_targets
    from tuj.m5_motion.scripted_grasps.objects.spoon import build_spoon_targets
    from tuj.m5_motion.scripted_grasps.registry import integration_status

    cases = (
        ("bread", "vac", "bread_vac", "bread_vac_"),
        ("fruit", "3F", None, "fruit_3f_"),
        ("spoon", "3F", None, "spoon_3f_c3_2_"),
        ("mug", "3F", "mug_c3_2", "mug_3f_"),
    )
    body = transform([0.35, -0.05, 0.92], rotation=np.eye(3))
    center = np.zeros(3)
    for object_id, ee, module_name, recipe_prefix in cases:
        entry = _c3_2_entry(object_id)
        assert entry.ee == ee
        assert entry.module_name == module_name
        assert integration_status(entry) == "EXPERIMENTAL"
        for suffix in ("_a", "_b"):
            instance = object_id + suffix
            resolved = resolve(request_for(entry, object_id=instance))
            assert resolved.object_id == object_id
            assert resolved.ee == ee
            assert resolved.scene_object_id == instance
            assert resolved.body_object_id == instance
            recipe = resolved.recipe()
            assert recipe.ee_id == ee
            assert recipe.recipe_id.startswith(recipe_prefix)
            if object_id == "spoon":
                targets = build_spoon_targets(body, center, recipe.expected_size_m, recipe)
            else:
                targets = build_catalog_targets(body, center, recipe.expected_size_m, recipe)
            assert set(targets) >= {"PRE_GRASP", "GRASP", "LIFT"}
            np.testing.assert_allclose(
                targets["LIFT"][:3, 3] - targets["GRASP"][:3, 3],
                [0, 0, recipe.lift_distance_m],
            )
    # Prior c2_1 / c1_2 exact-id recipes unchanged and remain validated.
    c2_bread = next(e for e in ENTRIES if e.object_id == "bread" and e.environment.startswith("C2_1"))
    c2_mug = next(e for e in ENTRIES if e.object_id == "mug" and e.environment.startswith("C2_1"))
    assert resolve(request_for(c2_bread)) == c2_bread
    assert resolve(request_for(c2_mug)) == c2_mug
    assert integration_status(c2_bread) == "VALIDATED"
    assert integration_status(c2_mug) == "VALIDATED"
    assert c2_bread.recipe().task_id == "c2_1" and c2_bread.recipe().ee_id == "3F"
    assert c2_mug.recipe().task_id == "c2_1" and c2_mug.recipe().ee_id == "3F"
    c1_spoon = next(e for e in ENTRIES if e.object_id == "spoon" and e.environment.startswith("C1_2"))
    assert resolve(request_for(c1_spoon)) == c1_spoon
    assert integration_status(c1_spoon) == "VALIDATED"
    assert c1_spoon.recipe().ee_id == "2F"
    assert c1_spoon.recipe().expected_size_m is None


def test_c3_2_plate_vac_rim_offset_is_geometry_derived():
    from tuj.m5_motion.scripted_grasps.objects import plate_vac
    recipe = plate_vac.plate_vac_recipe()
    half = 0.5 * min(recipe.expected_size_m[0], recipe.expected_size_m[1])
    radial = recipe.expected_size_m[0] * recipe.offset_fraction[0]
    assert radial == pytest.approx(
        half - plate_vac.CUP_CLEARANCE_RADIUS_M - plate_vac.RIM_EDGE_MARGIN_M)
    assert recipe.offset_fraction[2] == 0.5
    targets = plate_vac.build_plate_vac_targets(
        transform([0.4, 0.1, 0.9], rotation=np.eye(3)),
        np.zeros(3), recipe.expected_size_m, recipe)
    grasp_xy = targets["GRASP"][:2, 3]
    # Cup sits on a lateral rim patch, not at the plate center.
    assert np.linalg.norm(grasp_xy - np.array([0.4, 0.1])) == pytest.approx(radial, abs=1e-9)


def test_c3_2_bread_vac_does_not_seat_below_aabb_top():
    """Same vac cup ≡ grip-site rule as plate: no immersion past AABB top."""
    from tuj.m5_motion.scripted_grasps.objects import bread_vac
    assert bread_vac.SEATING_OFFSET_Z_M >= 0.0
    recipe = bread_vac.bread_vac_recipe()
    assert recipe.offset_m[2] >= 0.0
    assert recipe.offset_fraction[2] == 0.5
    size = np.asarray(recipe.expected_size_m, dtype=float)
    body_z = 0.92
    center = np.zeros(3)
    targets = bread_vac.build_bread_vac_targets(
        transform([0.40, -0.05, body_z], rotation=np.eye(3)),
        center, size, recipe)
    top_z = body_z + center[2] + 0.5 * size[2]
    assert targets["GRASP"][2, 3] >= top_z - 1e-12
    assert targets["GRASP"][2, 3] == pytest.approx(top_z + recipe.offset_m[2])


def test_vac_attach_rebind_restores_grasp_object_pose():
    """CLOSE-depressed pose must not define the kinematic attach transform."""
    from types import SimpleNamespace
    from dataclasses import dataclass
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        GRASP_RELATIVE_POSE_SOURCE,
        grasp_continuous_relative_pose,
        rebind_kinematic_attachment_to_object_pose,
    )
    from tuj.m5_motion.scripted_grasps.frames import inverse

    grip = transform([0.4, 0.0, 0.93], rotation=np.eye(3))
    grasp_object = transform([0.4, 0.0, 0.925], rotation=np.eye(3))
    crushed = transform([0.4, 0.01, 0.920], rotation=Rotation.from_euler('x', 8, degrees=True).as_matrix())
    grasp_T_GB = grasp_continuous_relative_pose(grip, grasp_object)
    assert np.allclose(grip @ grasp_T_GB, grasp_object)

    @dataclass(frozen=True)
    class _Attach:
        object_id: str = "plate_b"
        free_joint_name: str = "plate_b_joint0"
        reference_kind: str = "site"
        reference_name: str = "grip"
        position_in_reference_m: tuple = (0.0, 0.0, -0.006)
        rotation_in_reference: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        attach_distance_m: float = -0.0008
        mode: object = None
        breakable_weld: object = None

    applied = {}

    class _Runtime:
        def __init__(self):
            self._attachment = _Attach()
            self.attachment = self._attachment

        def synchronize_attached_object(self):
            rel = np.eye(4)
            rel[:3, 3] = self._attachment.position_in_reference_m
            rel[:3, :3] = np.asarray(self._attachment.rotation_in_reference)
            applied["body"] = grip @ rel
            self.attachment = self._attachment

    runtime = _Runtime()
    context = SimpleNamespace(
        runtime=runtime,
        grip_pose=lambda: grip,
        body_pose=lambda: crushed,
    )
    updated, rel = rebind_kinematic_attachment_to_object_pose(context, grasp_object)
    assert np.allclose(rel, grasp_T_GB)
    assert np.allclose(applied["body"], grasp_object)
    assert updated.position_in_reference_m == pytest.approx(tuple(grasp_T_GB[:3, 3]))
    # Crushed→grasp correction is a real discontinuity; LIFT must not inherit crush.
    assert np.linalg.norm(crushed[:3, 3] - grasp_object[:3, 3]) > 0.001
    assert GRASP_RELATIVE_POSE_SOURCE == "CONTROLLER_GRASP_OBJECT_POSE"
    assert np.allclose(inverse(grip) @ grasp_object, grasp_T_GB)


def test_vacuum_support_breakaway_skips_when_already_clear():
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import vacuum_support_breakaway_lift_m
    assert vacuum_support_breakaway_lift_m(0.0) == 0.0
    assert vacuum_support_breakaway_lift_m(0.01) == 0.0
    assert vacuum_support_breakaway_lift_m(1e-5) == 0.0


def test_vacuum_support_breakaway_lift_is_penetration_plus_pad():
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M,
        VACUUM_SUPPORT_CLEARANCE_PAD_M,
        vacuum_support_breakaway_lift_m,
    )
    # bread-like AABB immersion (~6.6 mm) — geometry-driven, not object-id based.
    clearance = -0.00659
    lift = vacuum_support_breakaway_lift_m(clearance)
    assert lift == pytest.approx(
        0.00659 + VACUUM_SUPPORT_CLEARANCE_PAD_M + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M
    )


def test_vacuum_support_breakaway_rejects_deep_invalid_immersion():
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        MAX_VACUUM_SUPPORT_BREAKAWAY_M,
        vacuum_support_breakaway_lift_m,
    )
    from tuj.m5_motion.scripted_grasps.runtime import GraspFailure
    with pytest.raises(GraspFailure, match='SUPPORT_BREAKAWAY_EXCEEDS_BOUND'):
        vacuum_support_breakaway_lift_m(-(MAX_VACUUM_SUPPORT_BREAKAWAY_M + 0.01))


def test_breakaway_vacuum_from_support_applies_bounded_cartesian_lift(tmp_path):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        VACUUM_SUPPORT_CLEARANCE_PAD_M,
        breakaway_vacuum_from_support,
    )

    grip = transform([0.5, -0.2, 0.97], rotation=Rotation.from_euler('x', 180, degrees=True).as_matrix())
    planned = {}

    class FakeData:
        def __init__(self):
            self.qpos = np.zeros(10)
            self.qvel = np.zeros(10)
            self.ncon = 0
            self.contact = []

    data = FakeData()
    arm_ids = np.array([0, 1, 2, 3, 4, 5])

    def plan_to(target, stage, cartesian=False):
        assert stage == 'BREAKAWAY'
        assert cartesian is True
        planned['target'] = np.asarray(target)
        return np.vstack([np.zeros(6), np.ones(6) * 0.01])

    def synchronize_attached_object():
        pass

    # Before: bread-like immersion; after kinematic path: pad+dip-margin clearance.
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M,
    )
    expected_lift = 0.00659 + VACUUM_SUPPORT_CLEARANCE_PAD_M + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M
    heights = [0.92 - 0.00659, 0.92 + VACUUM_SUPPORT_CLEARANCE_PAD_M + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M]

    context = SimpleNamespace(
        bottom_height=lambda: heights.pop(0) if heights else 0.92 + VACUUM_SUPPORT_CLEARANCE_PAD_M + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M,
        support_top_z=0.92,
        support_geom_names=lambda: ['island_island_group_top_2'],
        object_geoms=set(),
        grip_pose=lambda: grip,
        plan_to=plan_to,
        arm_ids=arm_ids,
        data=data,
        mj=SimpleNamespace(
            mj_fwdPosition=lambda *a, **k: None,
            mj_collision=lambda *a, **k: None,
        ),
        model=object(),
        runtime=SimpleNamespace(attachment=object(), synchronize_attached_object=synchronize_attached_object),
        object_dadr=6,
        robot=SimpleNamespace(_ref_joint_vel_indexes=[0, 1, 2, 3, 4, 5]),
        sample=lambda: None,
        step=lambda q, opening: None,
        output=tmp_path,
        stage=None,
    )
    q, record = breakaway_vacuum_from_support(context, np.ones(6), opening=1.0)
    assert record['applied'] is True
    assert record['mode'] == 'KINEMATIC_PATH'
    assert record['lift_m'] == pytest.approx(expected_lift)
    delta = planned['target'][:3, 3] - grip[:3, 3]
    assert delta[2] == pytest.approx(record['lift_m'], abs=1e-9)
    assert (tmp_path / 'vacuum_support_breakaway.json').exists()


def test_breakaway_vacuum_from_support_noop_when_clear(tmp_path):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import breakaway_vacuum_from_support

    def plan_to(*args, **kwargs):
        raise AssertionError('plan_to must not run when already clear')

    context = SimpleNamespace(
        bottom_height=lambda: 0.925,
        support_top_z=0.92,
        support_geom_names=lambda: ['island_island_group_top_2'],
        object_geoms=set(),
        data=SimpleNamespace(ncon=0, contact=[]),
        grip_pose=lambda: np.eye(4),
        plan_to=plan_to,
        output=tmp_path,
    )
    q_in = np.arange(6, dtype=float)
    q_out, record = breakaway_vacuum_from_support(context, q_in, opening=0.5)
    assert record['applied'] is False
    assert np.allclose(q_out, q_in)


def test_breakaway_vacuum_reports_stuck_when_clearance_does_not_improve(tmp_path):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import breakaway_vacuum_from_support
    from tuj.m5_motion.scripted_grasps.runtime import GraspFailure

    grip = transform([0.5, -0.2, 0.97], rotation=Rotation.from_euler('x', 180, degrees=True).as_matrix())

    class FakeData:
        qpos = np.zeros(10)
        qvel = np.zeros(10)
        ncon = 0
        contact = []

    context = SimpleNamespace(
        bottom_height=lambda: 0.92 - 0.005,
        support_top_z=0.92,
        support_geom_names=lambda: ['island_island_group_top_2'],
        object_geoms=set(),
        grip_pose=lambda: grip,
        plan_to=lambda *a, **k: np.zeros((2, 6)),
        arm_ids=np.arange(6),
        data=FakeData(),
        mj=SimpleNamespace(
            mj_fwdPosition=lambda *a, **k: None,
            mj_collision=lambda *a, **k: None,
        ),
        model=object(),
        runtime=SimpleNamespace(attachment=None),
        robot=SimpleNamespace(_ref_joint_vel_indexes=None),
        sample=lambda: None,
        step=lambda q, opening: None,
        output=tmp_path,
        stage=None,
    )
    with pytest.raises(GraspFailure, match='SUPPORT_BREAKAWAY_STUCK'):
        breakaway_vacuum_from_support(context, np.zeros(6), opening=-1.0)


def test_vacuum_support_breakaway_lift_uses_mesh_penetration_when_worse():
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M,
        VACUUM_SUPPORT_CLEARANCE_PAD_M,
        vacuum_support_breakaway_lift_m,
    )
    # AABB nearly flush, but mid-body mesh still penetrates (bread-like).
    lift = vacuum_support_breakaway_lift_m(-0.0005, mesh_penetration_m=0.004)
    assert lift == pytest.approx(
        0.004 + VACUUM_SUPPORT_CLEARANCE_PAD_M + VACUUM_POST_BREAKAWAY_LIFT_DIP_MARGIN_M
    )


def test_move_vacuum_cartesian_kinematic_plays_path_then_settles(tmp_path):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import move_vacuum_cartesian_kinematic

    class FakeData:
        def __init__(self):
            self.qpos = np.zeros(10)
            self.qvel = np.zeros(10)
            self.ncon = 0
            self.contact = []

    applied = []
    steps = []

    def plan_to(target, stage, cartesian=False):
        assert stage == 'LIFT'
        assert cartesian is True
        return np.vstack([np.zeros(6), np.arange(6, dtype=float) * 0.01])

    controller = SimpleNamespace(input_type='absolute', goal_qpos=None, update=lambda: None)
    context = SimpleNamespace(
        plan_to=plan_to,
        arm_ids=np.arange(6),
        data=FakeData(),
        mj=SimpleNamespace(
            mj_fwdPosition=lambda *a, **k: None,
            mj_collision=lambda *a, **k: None,
        ),
        model=object(),
        runtime=SimpleNamespace(attachment=None),
        robot=SimpleNamespace(
            _ref_joint_vel_indexes=[0, 1, 2, 3, 4, 5],
            part_controllers={'right': controller},
        ),
        sample=lambda: applied.append(context.data.qpos[:6].copy()),
        step=lambda q, opening: steps.append(np.asarray(q).copy()),
        stage=None,
        output=tmp_path,
    )
    q = move_vacuum_cartesian_kinematic(
        context, np.eye(4), 'LIFT', opening=-1.0, settle_steps=3)
    assert np.allclose(q, np.arange(6) * 0.01)
    assert len(applied) == 2
    assert len(steps) == 3
    assert np.allclose(controller.goal_qpos, q)


def test_settle_vacuum_arm_tracking_stops_when_converged():
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import settle_vacuum_arm_tracking

    class FakeData:
        def __init__(self):
            self.qpos = np.zeros(6)
            self.qvel = np.zeros(6)

    data = FakeData()
    steps = {'n': 0}

    def step(q, opening):
        steps['n'] += 1
        # Converge after a few commands.
        if steps['n'] >= 4:
            data.qpos[:] = q

    controller = SimpleNamespace(input_type='absolute', goal_qpos=None, update=lambda: None)
    context = SimpleNamespace(
        arm_ids=np.arange(6),
        data=data,
        robot=SimpleNamespace(part_controllers={'right': controller}),
        step=step,
    )
    q = np.arange(6, dtype=float) * 0.02
    err = settle_vacuum_arm_tracking(context, q, opening=-1.0, tol_rad=0.01, max_steps=20)
    assert err <= 0.01
    assert steps['n'] >= 6  # 4 to converge + 3 streak
    assert np.allclose(controller.goal_qpos, q)


def test_world_down_vacuum_grip_pose_levels_tool_axis_and_keeps_tcp():
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        vacuum_tool_tilt_from_world_down_rad,
        world_down_vacuum_grip_pose,
    )
    # ~7° tilt like live bread_b vac grasp.
    tilt = np.deg2rad(7.36)
    R = Rotation.from_rotvec([0.0, tilt, 0.0]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    grip = np.eye(4)
    grip[:3, :3] = R
    grip[:3, 3] = [2.4, -3.2, 1.11]
    assert vacuum_tool_tilt_from_world_down_rad(grip) == pytest.approx(tilt, abs=1e-6)
    leveled = world_down_vacuum_grip_pose(grip)
    assert vacuum_tool_tilt_from_world_down_rad(leveled) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(leveled[:3, 3], grip[:3, 3])
    assert np.allclose(leveled[:3, 2], [0.0, 0.0, -1.0])


def test_world_down_vacuum_grip_pose_preserves_held_body_origin():
    """Fixed-TCP level dropped live bread lift 0.125→0.090; body-origin mode must not."""
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        vacuum_tool_tilt_from_world_down_rad,
        world_down_vacuum_grip_pose,
    )
    from tuj.m5_motion.scripted_grasps.frames import inverse
    tilt = np.deg2rad(7.36)
    grip = np.eye(4)
    grip[:3, :3] = (
        Rotation.from_rotvec([0.0, tilt, 0.0]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    )
    grip[:3, 3] = [2.4, -3.2, 1.11]
    T_GB = np.eye(4)
    T_GB[:3, 3] = [0.0, 0.004, 0.027]  # bread-like cup seating
    body = grip @ T_GB
    leveled = world_down_vacuum_grip_pose(grip, body_pose=body)
    body_after = leveled @ T_GB
    assert vacuum_tool_tilt_from_world_down_rad(leveled) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(body_after[:3, 3], body[:3, 3], atol=1e-12)
    # TCP must move; fixed-TCP leveling is what sank lift_m on live HOLD.
    assert not np.allclose(leveled[:3, 3], grip[:3, 3])
    assert np.allclose(inverse(leveled) @ body_after, T_GB, atol=1e-12)


def test_run_kinematic_vacuum_hold_snaps_joints_each_tick():
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import run_kinematic_vacuum_hold

    class FakeData:
        def __init__(self):
            self.qpos = np.zeros(10)
            self.qvel = np.zeros(10)
            self.time = 0.0
            self.ncon = 0
            self.contact = []

    applied = []
    samples = []
    data = FakeData()

    def advance(action):
        data.time += 0.02
        # Impedance would sag; hold must overwrite after this tick.
        data.qpos[:6] = np.arange(6, dtype=float) * 0.5

    def apply_via_sample_path():
        # patched through _apply by monkeypatching module in test via context
        pass

    controller = SimpleNamespace(input_type='absolute', goal_qpos=None, update=lambda: None)
    context = SimpleNamespace(
        data=data,
        arm_ids=np.arange(6),
        robot=SimpleNamespace(
            action_dim=8,
            composite_controller=SimpleNamespace(
                _action_split_indexes={'right': (0, 6), 'right_gripper': (6, 8)}
            ),
            part_controllers={'right': controller},
            _ref_joint_vel_indexes=[0, 1, 2, 3, 4, 5],
        ),
        player=SimpleNamespace(_advance_controller=advance),
        mj=SimpleNamespace(
            mj_fwdPosition=lambda *a, **k: None,
            mj_collision=lambda *a, **k: None,
        ),
        model=object(),
        runtime=SimpleNamespace(attachment=None),
        sample=lambda: samples.append(data.qpos[:6].copy()) or {'lift_m': 0.12},
    )
    # Patch apply used inside hold
    import tuj.m5_motion.scripted_grasps.catalog_vacuum as vac

    def fake_apply(ctx, waypoint):
        applied.append(np.asarray(waypoint).copy())
        ctx.data.qpos[:6] = np.asarray(waypoint)

    original = vac._apply_vac_arm_waypoint
    vac._apply_vac_arm_waypoint = fake_apply
    try:
        q = np.arange(6, dtype=float) * 0.02
        rows, elapsed = run_kinematic_vacuum_hold(context, q, opening=1.0, duration_s=0.05)
    finally:
        vac._apply_vac_arm_waypoint = original
    assert elapsed == pytest.approx(0.06, abs=1e-9)
    assert len(rows) == 3
    assert all(np.allclose(a, q) for a in applied)
    assert all(np.allclose(s, q) for s in samples)
    assert np.allclose(controller.goal_qpos, q)


def test_straighten_vacuum_tool_world_down_skips_when_already_level(tmp_path):
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_vacuum import (
        straighten_vacuum_tool_world_down,
    )
    grip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(float)
    grip[:3, 3] = [1.0, 2.0, 3.0]
    context = SimpleNamespace(
        grip_pose=lambda: grip.copy(),
        output=tmp_path,
        plan_to=lambda *a, **k: (_ for _ in ()).throw(AssertionError('should skip')),
    )
    q_in = np.arange(6, dtype=float)
    q_out, record = straighten_vacuum_tool_world_down(context, q_in, opening=1.0)
    assert record['applied'] is False
    assert np.allclose(q_out, q_in)
    assert (tmp_path / 'vacuum_tool_level.json').exists()


def test_catalog_level_exempts_vac_cup_held_object_contact():
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_runtime import CatalogContext

    class FakeModel:
        def geom(self, g):
            names = {0: 'gripper0_right_vac_cup', 1: 'bread_b_g2', 2: 'island_island_group_top_2'}
            return SimpleNamespace(name=names[g])

    ctx = CatalogContext.__new__(CatalogContext)
    ctx.finger_geoms = {0}
    ctx.handle_geoms = {1}
    ctx.object_geoms = {1}
    ctx.model = FakeModel()
    ctx.support_released = True
    ctx.initial_body = np.eye(4)
    spoon_bad = [
        {'geoms': ['gripper0_right_vac_cup', 'bread_b_g2'], 'penetration_m': 0.001},
        {'geoms': ['island_island_group_top_2', 'bread_b_g2'], 'penetration_m': 0.005},
    ]

    def fake_super_bad(self, data, stage):
        return list(spoon_bad)

    import tuj.m5_motion.scripted_grasps.spoon_runtime as spoon_mod
    original = spoon_mod.SpoonContext.bad_contacts
    spoon_mod.SpoonContext.bad_contacts = fake_super_bad
    try:
        filtered = CatalogContext.bad_contacts(ctx, data=None, stage='LEVEL')
    finally:
        spoon_mod.SpoonContext.bad_contacts = original
    assert filtered == [spoon_bad[1]]


def test_catalog_breakaway_exempts_vac_cup_held_object_contact():
    """Plate vac cup↔plate contact must not fail PLAN_BREAKAWAY (live regression)."""
    from types import SimpleNamespace
    from tuj.m5_motion.scripted_grasps.catalog_runtime import CatalogContext

    class FakeModel:
        def geom(self, g):
            names = {0: 'gripper0_right_vac_cup', 1: 'plate_b_g1', 2: 'island_island_group_top_2'}
            return SimpleNamespace(name=names[g])

    ctx = CatalogContext.__new__(CatalogContext)
    ctx.finger_geoms = {0}
    ctx.handle_geoms = {1}
    ctx.object_geoms = {1}
    ctx.model = FakeModel()
    ctx.support_released = False
    ctx.initial_body = np.eye(4)

    spoon_bad = [
        {'geoms': ['gripper0_right_vac_cup', 'plate_b_g1'], 'penetration_m': 0.00221},
        {'geoms': ['island_island_group_top_2', 'plate_b_g1'], 'penetration_m': 0.005},
    ]

    def fake_super_bad(self, data, stage):
        assert stage == 'BREAKAWAY'
        return list(spoon_bad)

    import tuj.m5_motion.scripted_grasps.spoon_runtime as spoon_mod
    original = spoon_mod.SpoonContext.bad_contacts
    spoon_mod.SpoonContext.bad_contacts = fake_super_bad
    try:
        filtered = CatalogContext.bad_contacts(ctx, data=None, stage='BREAKAWAY')
    finally:
        spoon_mod.SpoonContext.bad_contacts = original

    assert filtered == [spoon_bad[1]]
    assert all('vac_cup' not in g for c in filtered for g in c['geoms'])


def test_early_lift_support_exemption_includes_island_top_1():
    """Resting plate may contact top_1 or top_2; early-LIFT exemption must cover both."""
    support_names = {"island_island_group_top_1", "island_island_group_top_2"}
    object_names = {"plate_b_g8", "plate_b_g26"}
    bad = [
        {
            "geoms": ["island_island_group_top_1", "plate_b_g8"],
            "penetration_m": 0.00111,
        },
        {
            "geoms": ["island_island_group_top_2", "plate_b_g26"],
            "penetration_m": 0.0009,
        },
        {"geoms": ["robot0_link", "wall"], "penetration_m": 0.002},
    ]
    filtered = [
        contact
        for contact in bad
        if not (
            any(name in contact["geoms"] for name in support_names)
            and any(name in object_names for name in contact["geoms"])
            and contact["penetration_m"] <= 0.002
        )
    ]
    assert filtered == [bad[2]]


def test_c3_2_plate_vac_does_not_seat_below_aabb_top():
    """Negative seating immerses the cup and presses the plate into the island.

    Catalog vac attaches at the actual contact pose; a depressed plate is then
    carried into LIFT with residual island penetration (~1 mm), which fails
    once support-contact exemption ends.
    """
    from tuj.m5_motion.scripted_grasps.objects import plate_vac
    assert plate_vac.SEATING_OFFSET_Z_M >= 0.0
    recipe = plate_vac.plate_vac_recipe()
    assert recipe.offset_m[2] >= 0.0
    size = np.asarray(recipe.expected_size_m, dtype=float)
    body_z = 0.92
    center = np.zeros(3)
    targets = plate_vac.build_plate_vac_targets(
        transform([0.46, 0.19, body_z], rotation=np.eye(3)),
        center, size, recipe)
    top_z = body_z + center[2] + 0.5 * size[2]
    # Grip site / cup face at or above the AABB top — never below.
    assert targets["GRASP"][2, 3] >= top_z - 1e-12
    assert targets["GRASP"][2, 3] == pytest.approx(top_z + recipe.offset_m[2])
    # LIFT only raises world Z; it must not be the source of downward seating.
    assert targets["LIFT"][2, 3] - targets["GRASP"][2, 3] == pytest.approx(
        recipe.lift_distance_m)


@pytest.mark.parametrize('environment',(
    'C1_2_DoughFlatten','C2_1_ObjectSorting','C3_1_ObjectSorting'))
@pytest.mark.parametrize('ee',('2F','3F'))
def test_spoon_dispatch_preserves_m4_selected_hand(environment,ee):
    entry=next(e for e in ENTRIES if (e.object_id,e.environment,e.ee)==(
        'spoon',environment,ee))
    request=request_for(entry)
    assert resolve(request)==entry
    recipe=entry.recipe()
    assert recipe.ee_id==ee
    assert recipe.model_class==(
        'JacoThreeFingerDexterousGripper' if ee=='3F'
        else 'Robotiq85Gripper')
    if ee=='3F' and environment.endswith('ObjectSorting'):
        assert recipe.lateral_offset_m==-.0035
        assert recipe.handle_fraction==-.08
        assert recipe.three_finger_hold_close_margin==.1


def test_three_finger_spoon_gate_requires_balanced_opposed_contact():
    from tuj.m5_motion.scripted_grasps.spoon_runtime import (
        three_finger_contact_established,three_finger_pinch_event,
        three_finger_ready)
    sample={'finger_contacts':['thumb','index','pinky'],
        'finger_force_n':{'thumb':3.,'index':1.5,'pinky':1.5},
        'normal_opposition':.9,'contact_span_m':.03}
    assert three_finger_contact_established(sample)
    assert three_finger_pinch_event(sample)
    overloaded={**sample,'finger_force_n':{'thumb':4.,'index':2.5,'pinky':2.}}
    assert three_finger_contact_established(overloaded)
    assert not three_finger_pinch_event(overloaded)
    one_sided={**sample,'finger_contacts':['thumb','index'],
        'finger_force_n':{'thumb':3.,'index':1.5,'pinky':0.}}
    assert not three_finger_contact_established(one_sided)
    ripple=[sample.copy() for _ in range(4)]+[{**sample,
        'finger_force_n':{'thumb':2.,'index':1.,'pinky':.5}}]
    assert three_finger_ready(ripple)
    assert not three_finger_ready(ripple[:-1]+[one_sided])


def test_three_finger_spoon_force_deadband_prevents_command_drift():
    from tuj.m5_motion.scripted_grasps.objects.spoon import spoon_recipe
    from tuj.m5_motion.scripted_grasps.spoon_runtime import update_three_finger_commands
    recipe=spoon_recipe('3F')
    command=np.array([-.42,-.41,-.41])
    np.testing.assert_allclose(update_three_finger_commands(
        command,[2.6,1.2,1.1],recipe),command)
    assert update_three_finger_commands(command,[5.,3.,3.],recipe)[0]>command[0]


def test_plate_is_routable_but_explicitly_experimental():
    plate = ENTRIES[0]
    assert plate.object_id == "plate"
    assert plate in ENABLED_ENTRIES
    assert plate.object_id in EXPERIMENTAL_INTEGRATION
    assert plate.object_id not in PENDING_INTEGRATION
    assert integration_status(plate) == "EXPERIMENTAL"
    assert resolve(request_for(plate)) == plate


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e.driver == "catalog"],
    ids=lambda e: f"{e.object_id}-{e.environment}-{e.ee}",
)
def test_catalog_targets_follow_body_translation_rotation_and_center_offset(entry):
    from tuj.m5_motion.scripted_grasps.catalog_types import build_catalog_targets
    recipe = entry.recipe()
    center = np.array([.012, -.003, .006])
    original = transform([.3, -.1, .8], rotation=np.eye(3))
    moved = transform([-.2, .4, .85], rotation=Rotation.from_euler("z", 37, degrees=True).as_matrix())
    a = build_catalog_targets(original, center, recipe.expected_size_m, recipe)
    b = build_catalog_targets(moved, center, recipe.expected_size_m, recipe)
    np.testing.assert_allclose(np.linalg.inv(original) @ a["GRASP"], np.linalg.inv(moved) @ b["GRASP"], atol=1e-12)
    np.testing.assert_allclose(b["T_WC"][:3, 3], moved[:3, 3] + moved[:3, :3] @ center)
    np.testing.assert_allclose(b["LIFT"][:3, 3] - b["GRASP"][:3, 3], [0, 0, recipe.lift_distance_m])


def test_collision_proxy_uses_observed_transform_without_runtime_attachment():
    request = request_for()
    held = AttachedObjectTransform(object_id="plate", free_joint_name="plate_free",
        reference_kind="site", reference_name="grip", position_in_reference_m=(.03, -.002, .01),
        orientation_in_reference_xyzw=(0, 0, 0, 1))
    request.world.robot_state.held_tool_id = "plate"
    request.world.metadata["contact_friction_held_objects"] = {"plate": held.model_dump()}
    request.world.objects = {"plate": {"pose": Pose(frame_id="world", position_m=(0, 0, 0), orientation_xyzw=(0, 0, 0, 1)).model_dump(), "free_joint_name": "plate_free"},
                             "other": {"pose": Pose(frame_id="world", position_m=(1, 0, 0), orientation_xyzw=(0, 0, 0, 1)).model_dump(), "free_joint_name": "other_free"}}
    factory = ToolUseJournalCollisionContextFactory(SimpleNamespace(model_version_for=lambda ee: "fixture"), attachment_reference_name="grip")
    context = factory._base_context(request, active_ee="2F")
    assert context.attached_object_transforms == [held]
    assert context.metadata["attachment_proxy"] == "CONTACT_FRICTION"
    assert request.world.robot_state.attached_object_id is None
    assert [p.object_id for p in context.free_object_poses] == ["other"]


def runtime_stub():
    runtime = ToolUseJournalEERuntime.__new__(ToolUseJournalEERuntime)
    runtime._closed = False
    runtime._env = SimpleNamespace(obj_body_id={"plate": 1})
    runtime._attachment = None
    runtime._active_ee = "2F"
    runtime._held_tool_id = None
    runtime._grasp_engaged = True
    runtime._gripper_command = 1.
    runtime.scripted_grasp_retention = SimpleNamespace(entry=ENTRIES[0])
    return runtime


def test_contact_resource_does_not_create_attachment_and_open_clears_retention():
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    assert runtime.attachment is None
    assert runtime.held_tool_id == "plate"
    runtime.command_gripper(engaged=False, suction=False)
    assert runtime.held_tool_id is None
    assert runtime.scripted_grasp_retention is None
    assert not runtime.grasp_engaged


def test_detach_event_physically_opens_contact_grasp():
    from tuj.m5_motion.schema import TrajectoryEvent
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
    event = TrajectoryEvent(event_id="release", event_type="DETACH_OBJECT", time_from_start_s=0., target_id="plate")
    player._execute_event(event)
    assert runtime.held_tool_id is None and runtime.gripper_command < 0.
    assert runtime.scripted_grasp_retention is None


def test_proxy_context_requires_matching_live_contact_resource():
    runtime = runtime_stub()
    runtime.mark_contact_friction_object_as_tool("plate")
    player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
    context = CollisionContext(context_id="held", active_ee="2F", attached_object_ids=["plate"],
        collision_model_version="fixture",
        metadata={"attachment_proxy": "CONTACT_FRICTION"})
    player._verify_runtime_context(context, label="start")
    runtime.scripted_grasp_retention = None
    with pytest.raises(ToolUseJournalRuntimeError, match="attached objects"):
        player._verify_runtime_context(context, label="start")


def test_proxy_context_matches_scene_instance_not_recipe_type_id():
    """C3_2 *_b contact holds: scene id must match, not recipe type id."""
    cases = (
        ("fruit", "fruit_b"),
        ("spoon", "spoon_b"),
        ("fork", "fork_b"),
        ("mug", "mug_b"),
    )
    for recipe_id, scene_id in cases:
        runtime = runtime_stub()
        runtime._env = SimpleNamespace(obj_body_id={scene_id: 1})
        runtime._active_ee = "3F"
        runtime.mark_contact_friction_object_as_tool(scene_id)
        runtime.scripted_grasp_retention = SimpleNamespace(
            entry=SimpleNamespace(
                object_id=recipe_id, scene_object_id=scene_id))
        player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
        context = CollisionContext(
            context_id="held", active_ee="3F",
            attached_object_ids=[scene_id],
            collision_model_version="fixture",
            metadata={"attachment_proxy": "CONTACT_FRICTION"})
        player._verify_runtime_context(context, label="plan start")


def test_detach_contact_grasp_matches_scene_instance_id():
    from tuj.m5_motion.schema import TrajectoryEvent
    cases = (
        ("fruit", "fruit_b"),
        ("spoon", "spoon_b"),
        ("fork", "fork_b"),
        ("mug", "mug_b"),
    )
    for recipe_id, scene_id in cases:
        runtime = runtime_stub()
        runtime._env = SimpleNamespace(obj_body_id={scene_id: 1})
        runtime.mark_contact_friction_object_as_tool(scene_id)
        runtime.scripted_grasp_retention = SimpleNamespace(
            entry=SimpleNamespace(
                object_id=recipe_id, scene_object_id=scene_id),
            close=lambda: None)
        player = ToolUseJournalKinematicTrajectoryPlayer(runtime)
        event = TrajectoryEvent(
            event_id="release", event_type="DETACH_OBJECT",
            time_from_start_s=0., target_id=scene_id)
        message = player._execute_event(event)
        assert f"contact-held {scene_id}" in message
        assert runtime.held_tool_id is None and runtime.gripper_command < 0.


def test_script_dispatch_is_lazy_and_next_request_receives_actual_joints(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[4])
    actual = request.world.model_copy(deep=True)
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: actual.model_copy(deep=True))
    calls = []
    def grasp(runtime, entry, output, **kwargs):
        calls.append(kwargs["request"].world.robot_state.joint_positions_rad)
        actual.robot_state.joint_positions_rad = [.37]
        actual.scene.signature = "measured-after-grasp"
        return {"status": "SUCCESS"}
    monkeypatch.setattr(context, "execute_grasp", grasp)
    def planner_factory(env, repository, **kwargs):
        def planner(next_request):
            assert next_request.world.robot_state.joint_positions_rad == [.37]
            assert next_request.world.scene.signature == "measured-after-grasp"
            raise RuntimeError("ordinary planner reached")
        return planner
    session = live.ScriptedGraspSession(SimpleNamespace(env=object()), tmp_path, tmp_path / "run", planner_factory=planner_factory)
    session.execute_request(request, completed_subgoal="pick")
    assert calls == [[0.]]
    request.task.action_type = "MOVE"
    with pytest.raises(RuntimeError, match="ordinary planner reached"):
        session.execute_request(request)
    assert [r["route"] for r in session.records] == ["SCRIPTED_GRASP", "M5_MOTION_PLAN"]


def test_failed_grasp_is_recorded_and_never_completes_subgoal(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[4])
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    def fail(*args, **kwargs):
        raise RuntimeError("CONTACT_NOT_STABLE")
    monkeypatch.setattr(context, "execute_grasp", fail)
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="CONTACT_NOT_STABLE"):
        session.execute_request(request, completed_subgoal="pick")
    assert session.world.scene.completed_subgoals == []
    manifest = json.loads((tmp_path / "live-execution-manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["steps"][0]["status"] == "FAILED"


def test_failed_experimental_plate_keeps_route_and_status(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for(ENTRIES[0])
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    monkeypatch.setattr(context, "execute_grasp",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("PRE_LIFT_CONTACT_NOT_STABLE")))
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match="PRE_LIFT_CONTACT_NOT_STABLE"):
        session.execute_request(request)
    step = json.loads((tmp_path / "live-execution-manifest.json").read_text())["steps"][0]
    assert step["route"] == "SCRIPTED_GRASP"
    assert step["object_id"] == "plate"
    assert step["integration_status"] == "EXPERIMENTAL"


def test_place_switches_contact_proxy_to_free_object_after_release():
    from tuj.m5_motion.tests.test_tool_use_journal_planning import _request, _artifact, _keyframe, _factory
    from tuj.m5_motion.schema import KeyframeType, KeyframeEventType
    held = AttachedObjectTransform(object_id="bottle", free_joint_name="bottle_free",
        reference_name="robot0_right_hand", position_in_reference_m=(.03, .01, -.02),
        orientation_in_reference_xyzw=(0, 0, 0, 1))
    target = Pose(frame_id="world", position_m=(.4, 0, .2), orientation_xyzw=(0, 0, 0, 1))
    request = _request(MotionGoal(goal_type="POSE", target_object_id="bottle", target_pose=target), action_type="PLACE")
    request.world.robot_state.held_tool_id = "bottle"
    request.world.metadata["contact_friction_held_objects"] = {"bottle": held.model_dump()}
    request.task.allowed_touch_objects = ["table_collision"]
    artifact = _artifact((
        _keyframe("transfer", KeyframeType.TRANSFER),
        _keyframe("place", KeyframeType.PLACE, events=(KeyframeEventType.DETACH_OBJECT, KeyframeEventType.GRIPPER_OPEN)),
        _keyframe("retreat", KeyframeType.RETREAT),
    ))
    setup = _factory().prepare(request, artifact)
    transfer, place, retreat = setup.keyframe_artifact.candidates[0].keyframes
    before = setup.collision_contexts[place.collision_context_id]
    after = setup.collision_contexts[retreat.collision_context_id]
    assert before.metadata["attachment_proxy"] == "CONTACT_FRICTION"
    assert before.attached_object_transforms == [held]
    assert ("bottle", "table_collision") in before.allowed_collision_pairs
    assert after.attached_object_ids == []
    assert next(p for p in after.free_object_poses if p.object_id == "bottle").pose == target


def test_experimental_plate_routes_to_grasp_without_planner_fallback(monkeypatch, tmp_path):
    from tuj.m5_motion.scripted_grasps import live, context
    request = request_for()
    monkeypatch.setattr(live, "snapshot", lambda runtime, previous=None: request.world.model_copy(deep=True))
    calls = []
    def grasp(*args, **kwargs):
        calls.append(kwargs["request"].task.goal.target_object_id)
        return {"status": "SUCCESS", "metrics": {}}
    def forbidden(*args, **kwargs):
        pytest.fail("experimental plate must not fall back to the ordinary planner")
    monkeypatch.setattr(context, "execute_grasp", grasp)
    session = live.ScriptedGraspSession(SimpleNamespace(), tmp_path, tmp_path, planner_factory=forbidden)
    record = session.execute_request(request)
    assert calls == ["plate"]
    assert record["route"] == "SCRIPTED_GRASP"
    assert record["integration_status"] == "EXPERIMENTAL"
