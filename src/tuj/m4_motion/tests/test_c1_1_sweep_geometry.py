"""C1_1 sweep frames use a held plate's broad face and live transform."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "c1_1_openai_motion_run.py"
    )
    spec = importlib.util.spec_from_file_location("c1_1_motion_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rotation(axis_xy: np.ndarray, roll_rad: float) -> np.ndarray:
    z_axis = np.asarray((*axis_xy, 0.0), dtype=float)
    x_axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    base = np.column_stack((x_axis, y_axis, z_axis))
    cosine, sine = np.cos(roll_rad), np.sin(roll_rad)
    return base @ np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def test_sweep_targets_are_ordered_by_observed_angular_continuity() -> None:
    module = _example_module()
    world = SimpleNamespace(
        objects={
            "zone": {"pose": {"position_m": [0.0, 0.0, 0.8]}},
            "negative": {"pose": {"position_m": [1.0, -1.0, 0.8]}},
            "center": {"pose": {"position_m": [2.0, 0.0, 0.8]}},
            "positive": {"pose": {"position_m": [1.0, 1.0, 0.8]}},
        }
    )

    ordered = module._angularly_order_sweep_targets(
        world,
        ("positive", "negative", "center", "positive"),
        "zone",
    )

    assert ordered == ["negative", "center", "positive"]


def test_radial_sweep_frames_use_broad_face_and_hover_transit() -> None:
    module = _example_module()
    attachment = (-0.075, 0.0, 0.010)
    world = SimpleNamespace(
        objects={
            "collection_zone_visual": {
                "pose": {"position_m": [-0.02, 0.0, 0.803]},
                "dimensions_m": [0.25, 0.18, 0.002],
            },
            "heavy_plate": {
                "pose": {"position_m": [-0.30, 0.17, 0.81]},
                "dimensions_m": [0.182, 0.182, 0.011],
            },
            "block_0": {
                "pose": {"position_m": [0.20, 0.12, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
            "block_1": {
                "pose": {"position_m": [0.24, -0.20, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
            "block_99": {
                "pose": {"position_m": [0.40, 0.30, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
        }
    )

    names = module._install_sweep_reference_frames(
        world,
        target_ids=("block_0", "block_1"),
        goal_region_id="collection_zone_visual",
        tool_id="heavy_plate",
        attachment_position_in_reference_m=attachment,
    )

    assert len(names) == 8
    for block_id in ("block_0", "block_1"):
        engage = world.objects[f"sweep_target_{block_id}_engage"]
        hover = world.objects[f"sweep_target_{block_id}_hover_start"]
        axis = np.asarray(engage["push_axis_world"][:2], dtype=float)
        rotation = _rotation(axis, float(engage["roll_rad"]))
        engage_eef = np.asarray(engage["pose"]["position_m"], dtype=float)
        hover_eef = np.asarray(hover["pose"]["position_m"], dtype=float)
        engage_plate_center = engage_eef + rotation @ np.asarray(attachment)
        hover_plate_center = hover_eef + rotation @ np.asarray(attachment)

        # Table surface is z=0.800 and the vertical plate radius is 0.091 m.
        assert np.isclose(engage_plate_center[2], 0.8925, atol=1e-9)
        assert np.isclose(
            hover_plate_center[2] - engage_plate_center[2], 0.10, atol=1e-9
        )
        assert engage["tool_target_position_m"] == pytest.approx(
            engage_plate_center
        )
        contact_offset = np.asarray(
            engage["plate_contact_offset_m"], dtype=float
        )
        contact_target = np.asarray(
            engage["contact_point_target_position_m"], dtype=float
        )
        block_position = np.asarray(
            world.objects[block_id]["pose"]["position_m"], dtype=float
        )
        expected_contact_xy = (
            block_position[:2]
            + axis * module._DEFAULT_MOTION_PROFILE.sweep_start_offset_m
        )
        assert contact_target[:2] == pytest.approx(expected_contact_xy)
        assert engage_plate_center + contact_offset == pytest.approx(
            contact_target, abs=1e-9
        )
        # The verified physical controller now uses rim contact and observes
        # block progress in closed loop rather than assuming an open-loop
        # broad-face displacement.
        assert engage["contact_surface"] == "RIM"
        assert np.linalg.norm(contact_offset) > 0.0
    assert not any("block_99" in name for name in names)


def test_live_grasp_transform_points_plate_broad_face_inward() -> None:
    module = _example_module()
    attachment_rotation = np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    )

    eef_rotation, tool_axis, _, _, face_sign, normal_deviation = (
        module._plate_broad_face_orientation(
            attachment_rotation_in_reference=attachment_rotation,
            outward_xy=np.asarray((1.0, 0.0)),
            variant_index=0,
            variant_count=5,
            normal_deviation_rad=0.0,
        )
    )

    plate_rotation = eef_rotation @ attachment_rotation
    face_normal = float(face_sign) * plate_rotation[:, 2]
    assert face_normal == pytest.approx((-1.0, 0.0, 0.0))
    assert normal_deviation == pytest.approx(0.0)
    assert np.isclose(np.linalg.norm(tool_axis), 1.0)


def test_retry_relaxations_never_turn_broad_face_edge_on() -> None:
    module = _example_module()
    profile = module._C1MotionProfile.from_mapping(
        {
            "sweep_plane_alignment_candidate_count": 16,
            "sweep_broad_face_max_normal_deviation_rad": np.pi / 4.0,
        }
    )

    deviations = [
        module._broad_face_normal_deviation_for_retry(profile, index)
        for index in range(5)
    ]
    assert deviations == pytest.approx(
        (0.0, np.pi / 8.0, -np.pi / 8.0, np.pi / 4.0, -np.pi / 4.0)
    )
    for deviation in deviations:
        for variant_index in range(16):
            eef_rotation, _, _, _, face_sign, actual_deviation = (
                module._plate_broad_face_orientation(
                    attachment_rotation_in_reference=np.eye(3),
                    outward_xy=np.asarray((1.0, 0.0)),
                    variant_index=variant_index,
                    variant_count=16,
                    normal_deviation_rad=deviation,
                )
            )
            face_normal = float(face_sign) * eef_rotation[:, 2]
            assert np.dot(face_normal, (-1.0, 0.0, 0.0)) >= np.cos(
                np.pi / 4.0
            ) - 1e-9
            assert actual_deviation == pytest.approx(deviation)


def _current_task_planner_selection() -> SimpleNamespace:
    pick = SimpleNamespace(
        candidate_id="pick-candidate",
        subgoal_id="SG1_s1_d1",
        action_type="acquire",
        mode=None,
        ee="2F",
        tool="light_plate",
        target_ids=[],
        goal_region_id=None,
        grasp=None,
    )
    target_groups = (
        ("SG1_s1_d2", tuple(f"block_{index}" for index in range(5))),
        ("SG1_s2_d2", tuple(f"block_{index}" for index in range(5, 10))),
        ("SG1_s3_d2", ("block_10", "block_11")),
    )
    sweeps = [
        SimpleNamespace(
            candidate_id=f"{subgoal_id}-candidate",
            subgoal_id=subgoal_id,
            action_type="tool_act",
            mode="sweep",
            ee="2F",
            tool="light_plate",
            target_ids=list(target_ids),
            goal_region_id="collection_zone_visual",
        )
        for subgoal_id, target_ids in target_groups
    ]
    steps = [
        SimpleNamespace(
            action="PICK_TOOL",
            kind="primitive",
            candidate_id=pick.candidate_id,
            subgoal_id=pick.subgoal_id,
            preconditions=[],
            postconditions=[],
        ),
        *[
            SimpleNamespace(
                action="EXECUTE_SUBGOAL",
                kind="subgoal",
                candidate_id=assignment.candidate_id,
                subgoal_id=assignment.subgoal_id,
                preconditions=[],
                postconditions=[],
            )
            for assignment in sweeps
        ],
    ]
    return SimpleNamespace(candidate_assignments=[pick, *sweeps], steps=steps)


def test_sweep_binding_reads_split_targets_from_current_m4_result() -> None:
    module = _example_module()
    binding = module._selected_sweep_binding(_current_task_planner_selection())

    assert binding.subgoal_id == "SG1_d2"
    assert binding.ee == "2F"
    assert binding.tool == "light_plate"
    assert binding.goal_region_id == "collection_zone_visual"
    assert binding.target_ids == tuple(f"block_{index}" for index in range(12))


def test_pick_binding_reads_selected_ee_and_tool_from_current_m4_result() -> None:
    module = _example_module()
    binding = module._selected_pick_binding(_current_task_planner_selection())

    assert binding.subgoal_id == "SG1_s1_d1"
    assert binding.ee == "2F"
    assert binding.tool == "light_plate"
    assert binding.target_ids == ("light_plate",)


def test_structured_m3_sweep_binding_does_not_need_condition_parsing() -> None:
    module = _example_module()
    assignment = SimpleNamespace(
        candidate_id="candidate-sweep",
        subgoal_id="sweep-objects",
        action_type="tool_act",
        mode="sweep",
        ee="arm-gripper",
        tool="selected-pusher",
        target_ids=["part-a", "part-b"],
        goal_region_id="goal-bin",
    )
    step = SimpleNamespace(
        kind="subgoal",
        candidate_id="candidate-sweep",
        subgoal_id="sweep-objects",
        preconditions=[],
        postconditions=[],
    )

    binding = module._selected_sweep_binding(
        SimpleNamespace(candidate_assignments=[assignment], steps=[step])
    )

    assert binding.action_type == "tool_act"
    assert binding.mode == "sweep"
    assert binding.ee == "arm-gripper"
    assert binding.tool == "selected-pusher"
    assert binding.target_ids == ("part-a", "part-b")
    assert binding.goal_region_id == "goal-bin"


def test_goal_check_uses_m3_selected_world_region_geometry() -> None:
    module = _example_module()
    positions = np.asarray(
        [
            [0.52, -0.19, 0.81],
            [0.70, -0.19, 0.81],
        ],
        dtype=float,
    )
    env = SimpleNamespace(
        obj_body_id={"part-a": 0, "part-b": 1},
        sim=SimpleNamespace(data=SimpleNamespace(_data=SimpleNamespace(xpos=positions))),
    )
    world = SimpleNamespace(
        objects={
            "goal-bin": {
                "pose": {"position_m": [0.50, -0.20, 0.80]},
                "dimensions_m": [0.10, 0.10, 0.02],
            }
        }
    )

    inside = module._inside_goal_region(
        env, world, ("part-a", "part-b"), "goal-bin"
    )

    assert inside == ["part-a"]


def test_last_micro_push_uses_remaining_full_block_goal_distance() -> None:
    module = _example_module()
    world = SimpleNamespace(
        objects={
            "goal-bin": {
                "pose": {"position_m": [-0.02, 0.0, 0.80]},
                "dimensions_m": [0.25, 0.18, 0.002],
            },
            "part-a": {
                "pose": {"position_m": [-0.02, -0.10, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
        }
    )

    distance = module._adaptive_micro_push_distance_m(
        world,
        block_id="part-a",
        goal_region_id="goal-bin",
        maximum_distance_m=0.03,
        inset_margin_m=0.003,
    )

    # Region y half-size 9 cm minus block half-size 1 cm leaves an 8 cm
    # center boundary. From y=-10 cm, 2 cm + 3 mm inset is sufficient.
    assert distance == pytest.approx(0.023)


def test_failed_contact_continuation_reduces_push_before_reacquiring() -> None:
    module = _example_module()

    first_retry = module._reduced_micro_push_limit_m(
        0.03,
        minimum_distance_m=0.0075,
        retry_scale=0.5,
    )
    second_retry = module._reduced_micro_push_limit_m(
        first_retry,
        minimum_distance_m=0.0075,
        retry_scale=0.5,
    )
    final_retry = module._reduced_micro_push_limit_m(
        second_retry,
        minimum_distance_m=0.0075,
        retry_scale=0.5,
    )

    assert first_retry == pytest.approx(0.015)
    assert second_retry == pytest.approx(0.0075)
    assert final_retry == pytest.approx(0.0075)


def test_cleanup_revisits_outside_and_support_unstable_blocks() -> None:
    module = _example_module()

    cleanup = module._cleanup_block_ids(
        ("inside", "outside", "lifted"),
        inside_ids=("inside", "lifted"),
        initial_positions_m={
            "inside": (0.0, 0.0, 0.8),
            "outside": (0.1, 0.0, 0.8),
            "lifted": (0.0, 0.1, 0.8),
        },
        current_positions_m={
            "inside": (0.0, 0.0, 0.8005),
            "outside": (0.2, 0.0, 0.8),
            "lifted": (0.0, 0.1, 0.804),
        },
        max_support_error_m=0.003,
    )

    assert cleanup == ["outside", "lifted"]


def test_motion_profile_accepts_known_overrides_and_rejects_unknown_fields() -> None:
    module = _example_module()

    profile = module._C1MotionProfile.from_mapping(
        {"hover_height_m": 0.15, "settle_required_consecutive_ticks": 7}
    )

    assert profile.hover_height_m == 0.15
    assert profile.settle_required_consecutive_ticks == 7
    with pytest.raises(ValueError, match="unknown C1 motion-profile fields"):
        module._C1MotionProfile.from_mapping({"mystery_value": 1.0})


def test_alignment_candidates_are_ranked_from_best_to_most_reachable() -> None:
    module = _example_module()
    profile = module._C1MotionProfile.from_mapping(
        {
            "sweep_plane_alignment_blend": 1.0,
            "sweep_plane_alignment_min_blend": 0.0,
            "sweep_plane_alignment_candidate_count": 5,
        }
    )

    assert module._sweep_alignment_blends(profile) == pytest.approx(
        (1.0, 0.75, 0.5, 0.25, 0.0)
    )
    assert module._nearest_alignment_variant_index(profile, 0.76) == 1
    assert module._nearest_alignment_variant_index(profile, 0.49) == 2


def test_micro_push_frame_advances_only_configured_observation_step() -> None:
    module = _example_module()
    world = SimpleNamespace(
        objects={
            "collection_zone_visual": {
                "pose": {"position_m": [-0.02, 0.0, 0.803]},
                "dimensions_m": [0.25, 0.18, 0.002],
            },
            "heavy_plate": {
                "pose": {"position_m": [-0.30, 0.17, 0.81]},
                "dimensions_m": [0.182, 0.182, 0.011],
            },
            "block_0": {
                "pose": {"position_m": [0.20, 0.12, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
        }
    )

    module._install_sweep_reference_frames(
        world,
        target_ids=("block_0",),
        goal_region_id="collection_zone_visual",
        tool_id="heavy_plate",
        attachment_position_in_reference_m=(-0.075, 0.0, 0.010),
        max_push_distance_m=0.03,
    )

    zone = np.asarray(
        world.objects["collection_zone_visual"]["pose"]["position_m"][:2]
    )
    block = np.asarray(world.objects["block_0"]["pose"]["position_m"][:2])
    end_contact = np.asarray(
        world.objects["sweep_target_block_0_end"][
            "contact_point_target_position_m"
        ][:2]
    )
    # max_push_distance limits the requested rim-contact observation step.
    # The block center is evaluated from MuJoCo feedback during execution.
    assert np.linalg.norm(block - zone) - np.linalg.norm(
        end_contact - zone
    ) == (
        pytest.approx(0.03)
    )
    assert world.objects["sweep_target_block_0_end"][
        "target_contact_height_offset_from_block_center_m"
    ] == pytest.approx(0.0)


def test_contact_continuation_preserves_live_eef_orientation() -> None:
    module = _example_module()
    world = SimpleNamespace(
        objects={
            "collection_zone_visual": {
                "pose": {"position_m": [-0.02, 0.0, 0.803]},
                "dimensions_m": [0.25, 0.18, 0.002],
            },
            "heavy_plate": {
                "pose": {"position_m": [-0.30, 0.17, 0.81]},
                "dimensions_m": [0.182, 0.182, 0.011],
            },
            "block_0": {
                "pose": {"position_m": [0.20, 0.12, 0.806]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
        }
    )
    eef_rotation = _rotation(np.asarray((0.6, -0.8)), 0.7)

    module._install_sweep_reference_frames(
        world,
        target_ids=("block_0",),
        goal_region_id="collection_zone_visual",
        tool_id="heavy_plate",
        attachment_position_in_reference_m=(-0.075, 0.0, 0.010),
        attachment_rotation_in_reference=np.eye(3),
        eef_rotation_override=eef_rotation,
    )

    frame = world.objects["sweep_target_block_0_end"]
    reconstructed = _rotation(
        np.asarray(frame["tool_axis_world"][:2], dtype=float),
        frame["tool_roll_rad"],
    )
    assert reconstructed == pytest.approx(eef_rotation, abs=1e-9)
    assert frame["orientation_source"] == "LIVE_EEF_CONTINUATION"


def test_recovery_lifts_before_backing_away_from_block() -> None:
    module = _example_module()
    world = SimpleNamespace(
        objects={
            "collection_zone_visual": {
                "pose": {"position_m": [-0.02, 0.0, 0.803]},
                "dimensions_m": [0.25, 0.18, 0.002],
            },
            "heavy_plate": {
                "pose": {"position_m": [-0.30, 0.17, 0.81]},
                "dimensions_m": [0.182, 0.182, 0.011],
            },
            "block_0": {
                "pose": {"position_m": [0.20, 0.12, 0.812]},
                "dimensions_m": [0.02, 0.02, 0.012],
            },
        }
    )
    eef_position = np.asarray((0.10, 0.20, 0.90))
    eef_rotation = _rotation(np.asarray((0.6, -0.8)), 0.7)

    module._install_sweep_reference_frames(
        world,
        target_ids=("block_0",),
        goal_region_id="collection_zone_visual",
        tool_id="heavy_plate",
        attachment_position_in_reference_m=(-0.075, 0.0, 0.010),
        attachment_rotation_in_reference=np.eye(3),
        eef_rotation_override=eef_rotation,
        recovery_eef_position_world_m=eef_position,
        recovery_lift_m=0.025,
        start_offset_m=0.02,
        table_surface_z_override_m=0.8,
    )

    lift = np.asarray(
        world.objects["sweep_target_block_0_recovery_lift"]["pose"][
            "position_m"
        ]
    )
    backoff = np.asarray(
        world.objects["sweep_target_block_0_recovery_backoff"]["pose"][
            "position_m"
        ]
    )
    engage = np.asarray(
        world.objects["sweep_target_block_0_engage"]["pose"]["position_m"]
    )
    assert lift == pytest.approx(eef_position + (0.0, 0.0, 0.025))
    assert backoff[2] - engage[2] == pytest.approx(0.025)
    assert world.objects["sweep_target_block_0_engage"][
        "target_contact_world_z_m"
    ] == pytest.approx(0.806)


def test_plate_vertical_extent_tracks_orientation_not_center_height() -> None:
    module = _example_module()
    dimensions = (0.182, 0.182, 0.011)

    horizontal = module._circular_plate_vertical_half_extent_m(
        np.eye(3), dimensions
    )
    vertical_rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    )
    vertical = module._circular_plate_vertical_half_extent_m(
        vertical_rotation, dimensions
    )

    assert horizontal == pytest.approx(0.0055)
    assert vertical == pytest.approx(0.091)


def test_broad_face_contact_height_clamps_to_real_plate_boundary() -> None:
    module = _example_module()
    vertical_rotation = np.asarray(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )

    offset = module._circular_plate_broad_face_contact_offset_m(
        vertical_rotation,
        (0.182, 0.182, 0.011),
        vertical_offset_m=-0.20,
        face_sign=1,
    )
    local = vertical_rotation.T @ offset

    assert np.linalg.norm(local[:2]) == pytest.approx(0.091)
    assert local[2] == pytest.approx(0.0055)


def test_rim_grasp_force_accounts_for_off_center_gravity_torque() -> None:
    module = _example_module()
    dimensions = (0.1818, 0.1818, 0.0111)

    light_force = module._rim_grasp_retention_force_n(
        mass_kg=0.2,
        sliding_friction=0.8,
        dimensions_m=dimensions,
        safety_factor=2.0,
        max_grip_force_n=235.0,
    )
    heavy_force = module._rim_grasp_retention_force_n(
        mass_kg=0.8,
        sliding_friction=0.8,
        dimensions_m=dimensions,
        safety_factor=2.0,
        max_grip_force_n=235.0,
    )

    assert light_force > 2.0 * 0.2 * 9.81 / 0.8
    assert heavy_force == pytest.approx(4.0 * light_force)
    assert heavy_force < 235.0


def test_contact_follow_ignores_intentional_rigid_hand_motion() -> None:
    module = _example_module()
    reference_rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    monitor = SimpleNamespace(
        _follow_object_to_reference_local_m=np.asarray((0.10, -0.02, -0.10)),
        samples=[
            {
                "object_position_m": [0.10, -0.02, -0.10],
                "grasp_reference_position_m": [0.0, 0.0, 0.0],
                "grasp_reference_rotation": np.eye(3),
                "contact_follow_active": True,
            },
            {
                "object_position_m": [1.02, 2.30, 0.9],
                "grasp_reference_position_m": [1.0, 2.20, 1.0],
                "grasp_reference_rotation": reference_rotation,
                "contact_follow_active": True,
            }
        ],
        _bilateral=lambda sample: True,
        contact_follow_gain=0.5,
        contact_follow_max_m=0.04,
        contact_follow_max_tick_m=0.002,
    )

    correction = module._PhysicalGraspMonitor.contact_follow_translation_xy_m(
        monitor
    )

    assert correction == pytest.approx((0.0, 0.0), abs=1e-12)


def test_contact_follow_responds_only_to_relative_plate_motion() -> None:
    module = _example_module()
    monitor = SimpleNamespace(
        _follow_object_to_reference_local_m=np.asarray((0.10, -0.02, -0.10)),
        samples=[
            {
                "object_position_m": [0.10, -0.02, -0.10],
                "grasp_reference_position_m": [0.0, 0.0, 0.0],
                "grasp_reference_rotation": np.eye(3),
                "contact_follow_active": True,
            },
            {
                "object_position_m": [1.11, 2.18, 0.9],
                "grasp_reference_position_m": [1.0, 2.20, 1.0],
                "grasp_reference_rotation": np.eye(3),
                "contact_follow_active": True,
            }
        ],
        _bilateral=lambda sample: True,
        contact_follow_gain=0.5,
        contact_follow_max_m=0.04,
        contact_follow_max_tick_m=0.002,
    )

    correction = module._PhysicalGraspMonitor.contact_follow_translation_xy_m(
        monitor
    )

    # The raw -5 mm damping request is capped to the configured 2 mm/tick.
    assert correction == pytest.approx((-0.002, 0.0), abs=1e-12)


def test_new_plan_inherits_regrasp_roll_as_its_zero_delta_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _example_module()
    monkeypatch.setattr(
        module.ToolUseJournalControllerTrajectoryPlayer,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monitor = SimpleNamespace(current_regrasp_roll_offset_rad=-np.pi / 2.0)

    player = module._PhysicalGraspControllerTrajectoryPlayer(
        object(), monitor=monitor
    )

    assert player._regrasp_roll_offset_rad == pytest.approx(-np.pi / 2.0)
    assert player._plan_regrasp_roll_baseline_rad == pytest.approx(-np.pi / 2.0)
    assert (
        player._regrasp_roll_offset_rad
        - player._plan_regrasp_roll_baseline_rad
    ) == pytest.approx(0.0)


def test_contact_follow_does_not_restore_a_static_new_equilibrium() -> None:
    module = _example_module()
    monitor = SimpleNamespace(
        _follow_object_to_reference_local_m=np.asarray((0.10, -0.02, -0.10)),
        samples=[
            {
                "object_position_m": [0.06, -0.01, -0.08],
                "grasp_reference_position_m": [0.0, 0.0, 0.0],
                "grasp_reference_rotation": np.eye(3),
                "contact_follow_active": True,
            },
            {
                "object_position_m": [1.06, 1.99, 0.92],
                "grasp_reference_position_m": [1.0, 2.0, 1.0],
                "grasp_reference_rotation": np.eye(3),
                "contact_follow_active": True,
            },
        ],
        _bilateral=lambda sample: True,
        contact_follow_gain=0.5,
        contact_follow_max_m=0.04,
        contact_follow_max_tick_m=0.002,
    )

    correction = module._PhysicalGraspMonitor.contact_follow_translation_xy_m(
        monitor
    )

    # The plate no longer matches its original activation pose, but it is not
    # moving relative to the hand, so velocity damping must apply no force.
    assert correction == pytest.approx((0.0, 0.0), abs=1e-12)


def test_physical_tool_settle_uses_live_plate_not_rigid_eef_error() -> None:
    module = _example_module()
    monitor = SimpleNamespace(
        live_tool_observation=lambda: {
            "position_m": [0.201, -0.099, 0.881],
            "bottom_clearance_m": 0.0012,
            "linear_speed_m_s": 0.004,
            "stable_bilateral_contact": True,
            "contact_count": 4,
            "normal_force_n": 60.0,
            "target_block_id": "part-a",
        }
    )
    player = SimpleNamespace(
        _physical_grasp_monitor=monitor,
        _clearance_offset_m=0.007,
    )
    segment = SimpleNamespace(
        metadata={
            "physical_tool_settle": {
                "target_position_m": [0.20, -0.10, 0.8815],
                "target_clearance_m": 0.0015,
                "xy_tolerance_m": 0.015,
                "clearance_tolerance_m": 0.003,
                "max_table_penetration_m": 0.001,
                "max_tool_speed_m_s": 0.02,
            }
        }
    )

    result = module._PhysicalGraspControllerTrajectoryPlayer._custom_settle_evaluation(
        player,
        segment=segment,
        settle_config={"joint_tolerance_rad": 0.02},
        joint_error_rad=0.05,
        eef_position_error_m=0.03,
    )

    assert result is not None
    assert result["succeeded"] is True
    assert result["joint_ok"] is False
    assert result["tool_xy_ok"] is True
    assert result["clearance_ok"] is True
    # A large rigid EEF error is diagnostic only for a free frictional tool.
    assert result["eef_position_error_m"] == pytest.approx(0.03)


def test_tool_clearance_summary_is_independent_of_grasp_retention() -> None:
    module = _example_module()
    control = {
        "target_clearance_m": 0.0015,
        "clearance_tolerance_m": 0.003,
        "max_table_penetration_m": 0.001,
    }
    monitor = SimpleNamespace(
        samples=[
            {"bottom_clearance_m": 0.001, "physical_tool_control": control},
            {"bottom_clearance_m": -0.0005, "physical_tool_control": control},
        ]
    )

    result = module._PhysicalGraspMonitor.tool_clearance_summary(monitor)

    assert result["status"] == "SUCCESS"
    assert result["maximum_table_penetration_m"] == pytest.approx(0.0005)


def test_push_playback_pauses_until_contact_is_reacquired() -> None:
    module = _example_module()
    monitor = SimpleNamespace(
        set_active_segment=lambda segment: None,
        active_physical_push_control={
            "contact_plan_time_scale": 0.5,
            "reacquire_timeout_s": 1.5,
            "max_reacquire_attempts": 2,
        },
        push_contact_acquired=True,
        push_contact_loss_started_time_s=None,
        push_reacquire_attempts=0,
        push_recovery_exhausted=False,
        push_recovery_events=[],
        active_segment_id="push",
        samples=[
            {
                "target_block_contact_count": 0,
                "simulation_time_s": 10.0,
            }
        ],
    )
    player = SimpleNamespace(_physical_grasp_monitor=monitor)
    segment = SimpleNamespace()

    paused = module._PhysicalGraspControllerTrajectoryPlayer._plan_time_step_s(
        player,
        segment=segment,
        control_timestep_s=0.02,
    )
    monitor.samples[-1]["target_block_contact_count"] = 1
    advancing = (
        module._PhysicalGraspControllerTrajectoryPlayer._plan_time_step_s(
            player,
            segment=segment,
            control_timestep_s=0.02,
        )
    )

    assert paused == 0.0
    assert advancing == pytest.approx(0.01)
    assert monitor.push_recovery_events[-1]["status"] == "REACQUIRED"


def test_push_playback_stops_waiting_after_reacquire_timeout() -> None:
    module = _example_module()
    monitor = SimpleNamespace(
        set_active_segment=lambda segment: None,
        active_physical_push_control={
            "contact_plan_time_scale": 0.5,
            "reacquire_timeout_s": 1.5,
            "max_reacquire_attempts": 2,
        },
        push_contact_acquired=True,
        push_contact_loss_started_time_s=10.0,
        push_reacquire_attempts=1,
        push_recovery_exhausted=False,
        push_recovery_events=[{"attempt": 1}],
        active_segment_id="push",
        samples=[
            {
                "target_block_contact_count": 0,
                "simulation_time_s": 11.5,
            }
        ],
    )
    player = SimpleNamespace(_physical_grasp_monitor=monitor)

    step = module._PhysicalGraspControllerTrajectoryPlayer._plan_time_step_s(
        player,
        segment=SimpleNamespace(),
        control_timestep_s=0.02,
    )

    assert step == pytest.approx(0.02)
    assert monitor.push_recovery_exhausted is True
    assert monitor.push_recovery_events[-1]["status"] == "EXHAUSTED"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"execution_succeeded": False}, "EXECUTION_FAILED"),
        ({"block_support_lift_m": 0.0031}, "EXCESSIVE_BLOCK_LIFT"),
        ({"target_contact_sample_count": 0}, "NO_TARGET_CONTACT"),
        ({"radial_progress_m": -0.001}, "INSUFFICIENT_GOAL_PROGRESS"),
        ({"radial_progress_m": 0.0049}, "INSUFFICIENT_GOAL_PROGRESS"),
        ({}, None),
    ),
)
def test_physical_push_validation_rejects_uncommittable_state(
    overrides: dict[str, object], expected: str | None
) -> None:
    module = _example_module()
    inputs = {
        "execution_succeeded": True,
        "reached_goal": False,
        "target_contact_sample_count": 1,
        "radial_progress_m": 0.005,
        "minimum_progress_m": 0.005,
        "block_support_lift_m": 0.0,
        "maximum_block_lift_m": 0.003,
    }
    inputs.update(overrides)

    assert module._physical_push_rejection_reason(**inputs) == expected


def test_physical_push_validation_accepts_stable_goal_despite_small_progress() -> None:
    module = _example_module()

    reason = module._physical_push_rejection_reason(
        execution_succeeded=True,
        reached_goal=True,
        target_contact_sample_count=0,
        radial_progress_m=0.0,
        minimum_progress_m=0.005,
        block_support_lift_m=0.001,
        maximum_block_lift_m=0.003,
    )

    assert reason is None


def test_task_geometry_binder_prevents_vlm_from_perturbing_grounded_pose() -> None:
    module = _example_module()
    request = SimpleNamespace(
        world=SimpleNamespace(
            objects={
                "grounded_hover": {
                    "reference_frame_kind": "TASK_GEOMETRY",
                    "tool_axis_world": [0.0, 3.0, 4.0],
                    "tool_roll_rad": -0.75,
                },
                "grounded_end": {
                    "reference_frame_kind": "TASK_GEOMETRY",
                    "tool_axis_world": [0.0, 3.0, 4.0],
                    "tool_roll_rad": -0.75,
                },
            }
        )
    )
    strategy_provenance = module.StrategyGenerationProvenance(
        generator_kind=module.StrategyGeneratorKind.TASK_GEOMETRY,
        generator_id="test",
        input_hash="input",
    )
    raw = module.KeyframePlanArtifact(
        artifact_id="raw",
        provenance=module._provenance(
            "raw-artifact",
            "KeyframePlanArtifact",
            module.ModuleName.MOTION_PLANNER,
        ),
        scene_signature="scene",
        subgoal_id="subgoal",
        candidates=[
            module.KeyframePlanCandidate(
                strategy_id="vlm-choice",
                provenance=strategy_provenance,
                keyframes=[
                    module.RelativeKeyframeSpec(
                        keyframe_id="hover",
                        keyframe_type=module.KeyframeType.TRANSFER,
                        frame_ref="object:grounded_hover",
                        anchor="top",
                        approach_axis_xyz=(1.0, 0.0, 0.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.12,
                        roll_rad=1.25,
                        planner=module.KeyframePlannerType.SAMPLING_BASED,
                    ),
                    module.RelativeKeyframeSpec(
                        keyframe_id="end",
                        keyframe_type=module.KeyframeType.CUSTOM,
                        frame_ref="object:grounded_end",
                        anchor="top",
                        approach_axis_xyz=(1.0, 0.0, 0.0),
                        tool_axis_to_align="-z",
                        offset_along_approach_m=0.08,
                        roll_rad=0.5,
                        planner=module.KeyframePlannerType.CARTESIAN,
                    ),
                ],
            )
        ],
    )

    bound = module._bind_grounded_task_geometry_keyframes(raw, request)

    for keyframe in bound.candidates[0].keyframes:
        assert keyframe.anchor == "center"
        assert keyframe.approach_axis_xyz == pytest.approx((0.0, 0.6, 0.8))
        assert keyframe.tool_axis_to_align == "+z"
        assert keyframe.offset_along_approach_m == 0.0
        assert keyframe.roll_rad == pytest.approx(-0.75)
        assert (
            keyframe.metadata["metric_grounding_source"]
            == "TASK_GEOMETRY_REFERENCE_FRAME"
        )
    assert (
        bound.provenance.metadata["geometry_binder"]
        == "TASK_GEOMETRY_KEYFRAME_BINDER_V1"
    )
    assert bound.provenance.metadata["bound_frame_count"] == 2


def test_grounded_sweep_end_frame_supplies_physical_contact_control() -> None:
    module = _example_module()
    profile = module._DEFAULT_MOTION_PROFILE
    frame = {
        "target_block_id": "block_0",
        "push_axis_world": [1.0, 0.0, 0.0],
        "plate_contact_offset_local_m": [-0.08, 0.0, 0.005],
        "target_block_support_m": 0.01,
        "target_contact_height_offset_from_block_center_m": 0.0,
        "target_contact_world_z_m": 0.806,
        "block_support_center_z_m": 0.806,
        "tool_target_position_m": [0.1, 0.2, 0.9],
    }

    metadata = module._sweep_frame_execution_metadata(
        "sweep_target_block_0_end",
        frame,
        profile=profile,
    )

    assert metadata["target_block_id"] == "block_0"
    assert metadata["physical_tool_control"]["target_clearance_m"] == pytest.approx(
        profile.plate_table_clearance_m
    )
    push = metadata["physical_push_control"]
    assert push["push_axis_world"] == [1.0, 0.0, 0.0]
    assert push["contact_offset_local_m"] == [-0.08, 0.0, 0.005]
    assert push["contact_height_target_m"] == pytest.approx(0.806)
    assert metadata["physical_tool_target_position_m"] == [0.1, 0.2, 0.9]


def test_vlm_route_expands_over_all_grounded_orientation_variants() -> None:
    module = _example_module()
    objects = {}
    for variant_index in (0, 1):
        for phase in ("hover_start", "end"):
            name = f"variant_{variant_index}_block_0_{phase}"
            objects[name] = {
                "reference_frame_kind": "TASK_GEOMETRY",
                "orientation_variant_index": variant_index,
                "target_block_id": "block_0",
            }
    request = SimpleNamespace(world=SimpleNamespace(objects=objects))
    raw = module.KeyframePlanArtifact(
        artifact_id="raw",
        provenance=module._provenance(
            "raw-artifact",
            "KeyframePlanArtifact",
            module.ModuleName.MOTION_PLANNER,
        ),
        scene_signature="scene",
        subgoal_id="subgoal",
        candidates=[
            module.KeyframePlanCandidate(
                strategy_id="vlm-route",
                provenance=module.StrategyGenerationProvenance(
                    generator_kind=module.StrategyGeneratorKind.TASK_GEOMETRY,
                    generator_id="test",
                    input_hash="input",
                ),
                keyframes=[
                    module.RelativeKeyframeSpec(
                        keyframe_id="hover",
                        keyframe_type=module.KeyframeType.TRANSFER,
                        frame_ref="object:variant_0_block_0_hover_start",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        planner=module.KeyframePlannerType.SAMPLING_BASED,
                    ),
                    module.RelativeKeyframeSpec(
                        keyframe_id="end",
                        keyframe_type=module.KeyframeType.CUSTOM,
                        frame_ref="object:variant_0_block_0_end",
                        anchor="center",
                        approach_axis_xyz=(0.0, 0.0, 1.0),
                        planner=module.KeyframePlannerType.CARTESIAN,
                    ),
                ],
            )
        ],
    )

    expanded = module._expand_task_geometry_orientation_variants(raw, request)

    assert len(expanded.candidates) == 2
    assert [
        candidate.metadata["grounded_orientation_variant_index"]
        for candidate in expanded.candidates
    ] == [0, 1]
    for variant_index, candidate in enumerate(expanded.candidates):
        assert all(
            f"object:variant_{variant_index}_block_0_" in keyframe.frame_ref
            for keyframe in candidate.keyframes
        )
    assert (
        expanded.provenance.metadata["orientation_expander"]
        == "TASK_GEOMETRY_ORIENTATION_EXPANDER_V1"
    )


def test_video_transaction_commits_only_accepted_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _example_module()

    class _FakeWriter:
        def __init__(self) -> None:
            self.frames: list[np.ndarray] = []
            self.released = False

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())

        def release(self) -> None:
            self.released = True

    writer = _FakeWriter()
    monkeypatch.setattr(module.cv2, "VideoWriter", lambda *args, **kwargs: writer)
    env = SimpleNamespace(
        control_timestep=0.1,
        sim=SimpleNamespace(
            render=lambda **kwargs: np.zeros((8, 16, 3), dtype=np.uint8)
        ),
    )
    recorder = module._OffscreenVideoRecorder(
        env,
        tmp_path / "transaction.mp4",
        camera="fixed",
        width=16,
        height=8,
        fps=5.0,
    )
    assert recorder.frame_count == 1

    recorder.begin_transaction()
    recorder.write_current_frame()
    recorder.write_current_frame()
    assert recorder.rollback_transaction() == 2
    assert recorder.frame_count == 1
    assert len(writer.frames) == 1

    recorder.begin_transaction()
    recorder.write_current_frame()
    assert recorder.commit_transaction() == 1
    assert recorder.frame_count == 2
    assert len(writer.frames) == 2

    recorder.close()
    assert writer.released is True
