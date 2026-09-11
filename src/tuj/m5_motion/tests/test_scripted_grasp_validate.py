"""Unit tests for scripted grasp validation helpers (no MuJoCo / LLM)."""
import numpy as np
import pytest

from tuj.m5_motion.scripted_grasps.frames import transform
from tuj.m5_motion.scripted_grasps.validate import (
    C3_2_EXPECTED_ACQUIRES,
    attachment_discontinuity_m,
    batch_static_c3_2,
    format_batch_table,
    resolve_case,
    static_validate_case,
)


@pytest.mark.parametrize("instance", ["bread_b", "spoon_b", "fork_b"])
def test_c3_2_validator_can_resolve_type_level_2f_experiments(instance):
    resolved = resolve_case(instance, "2F")
    assert resolved["status"] == "RESOLVED"
    assert resolved["entry"].scene_object_id == instance
    assert resolved["entry"].object_id == instance[:-2]


@pytest.mark.parametrize(
    ("instance", "ee"),
    [("bread_b", "vac"), ("spoon_b", "3F"), ("fork_b", "3F")],
)
def test_c3_2_primary_registrations_remain_available(instance, ee):
    resolved = resolve_case(instance, ee)
    assert resolved["status"] == "RESOLVED"
    assert resolved["entry"].ee == ee


def test_c3_2_2f_experiments_are_not_visible_to_normal_m5_resolution():
    from tuj.m5_motion.scripted_grasps.registry import resolve
    from tuj.m5_motion.scripted_grasps.validate import _acquire_request

    request = _acquire_request("C3_2_BreakfastTrayPreparation", "bread_b", "2F")
    request.task.metadata.pop("scripted_grasp_validator_experimental")
    with pytest.raises(ValueError, match="SCRIPTED_GRASP_EE_MISMATCH"):
        resolve(request)


@pytest.mark.parametrize("instance", ["plate_a", "plate_b"])
def test_plate_instances_resolve_and_bind_independently(instance):
    resolved = resolve_case(instance, "vac")
    assert resolved["status"] == "RESOLVED"
    entry = resolved["entry"]
    assert entry.object_id == "plate"
    assert entry.ee == "vac"
    assert entry.module_name == "plate_vac"
    assert entry.scene_object_id == instance
    assert entry.body_object_id == instance
    report = static_validate_case(instance, "vac")
    assert report["static_status"] == "PASS"
    assert report["scene_object_id"] == instance
    assert set(report["targets"]) >= {"PRE_GRASP", "GRASP", "LIFT"}


@pytest.mark.parametrize("instance", ["fork_a", "fork_b"])
def test_fork_instances_resolve_and_bind_independently(instance):
    resolved = resolve_case(instance, "3F")
    assert resolved["status"] == "RESOLVED"
    entry = resolved["entry"]
    assert entry.object_id == "fork"
    assert entry.scene_object_id == instance
    report = static_validate_case(instance, "3F")
    assert report["static_status"] == "PASS"
    assert report["body_object_id"] == instance


def test_recipe_ee_mismatch_is_detected():
    resolved = resolve_case("plate_b", "3F")
    assert resolved["status"] == "EE_MISMATCH"
    report = static_validate_case("plate_b", "3F")
    assert report["static_status"] == "EE_MISMATCH"


def test_not_registered_is_reported_without_inventing_recipe():
    resolved = resolve_case("tongs_a", "3F")
    assert resolved["status"] == "NOT_REGISTERED"
    report = static_validate_case("tongs_a", "3F")
    assert report["static_status"] == "NOT_REGISTERED"
    assert report.get("recipe_id") is None


def test_support_penetration_from_negative_vac_seating_is_detected():
    # Synthetic thin body whose commanded vac TCP sits below the AABB top.
    from tuj.m5_motion.scripted_grasps import validate as validate_mod
    from tuj.m5_motion.scripted_grasps.objects import plate_vac

    original = plate_vac.SEATING_OFFSET_Z_M
    try:
        plate_vac.SEATING_OFFSET_Z_M = -0.002
        # Rebuild recipe path uses module constant inside plate_vac_recipe.
        size = np.asarray(plate_vac.PLATE_EXPECTED_SIZE_M)
        body = transform([0.46, 0.19, 0.92], rotation=np.eye(3))
        report = static_validate_case(
            "plate_b",
            "vac",
            T_WB=body,
            center_in_body_m=np.zeros(3),
            local_size_m=size,
            support_top_z=0.92 - 0.5 * size[2],  # flush support under AABB bottom
        )
        # Force rebuild with monkeypatched constant by calling recipe factory after patch.
        recipe = plate_vac.plate_vac_recipe()
        assert recipe.offset_m[2] == pytest.approx(-0.002)
        targets, _ = validate_mod.build_targets_for_entry(
            resolve_case("plate_b", "vac")["entry"], body, np.zeros(3), size,
        )
        # Re-run checks using the patched recipe geometry directly.
        top = 0.92 + 0.5 * size[2]
        assert targets["GRASP"][2, 3] < top
        report = static_validate_case(
            "plate_b",
            "vac",
            T_WB=body,
            center_in_body_m=np.zeros(3),
            local_size_m=size,
            support_top_z=0.92 - 0.5 * size[2],
        )
        # static_validate_case calls plate_vac_recipe() which reads the patched constant.
        assert "GRASP_SEATS_BELOW_AABB_TOP" in report["failures"]
        assert "GRASP_MAY_PRESS_OBJECT_INTO_SUPPORT" in report["failures"]
        assert report["static_status"] == "FAIL"
    finally:
        plate_vac.SEATING_OFFSET_Z_M = original


def test_attachment_discontinuity_detection():
    before = transform([0.1, 0.2, 0.3], rotation=np.eye(3))
    after = transform([0.1, 0.2, 0.305], rotation=np.eye(3))
    jump = attachment_discontinuity_m(before, after)
    assert jump == pytest.approx(0.005)
    assert jump > 0.001


def test_controller_displacement_is_not_mislabeled_as_attach_jump():
    """A successful commanded lift is distinct from the attach transition."""
    before = transform([0.1, 0.2, 0.3], rotation=np.eye(3))
    after_lift = transform([0.1, 0.2, 0.48], rotation=np.eye(3))
    assert attachment_discontinuity_m(before, after_lift) == pytest.approx(0.18)


def test_batch_summary_covers_all_c3_2_acquires():
    result = batch_static_c3_2()
    assert len(result["cases"]) == len(C3_2_EXPECTED_ACQUIRES)
    assert result["not_registered_count"] == 0
    assert result["ee_mismatch_count"] == 0
    table = format_batch_table(result["cases"])
    assert "plate_a" in table and "fork_b" in table and "mug_b" in table
    assert "NOT_RUN" in table
    by_id = {case["object_id"]: case for case in result["cases"]}
    assert by_id["plate_a"]["static_status"] == "PASS"
    assert by_id["fork_b"]["static_status"] == "PASS"
    assert by_id["bread_a"]["static_status"] == "PASS"
    assert by_id["bread_b"]["static_status"] == "PASS"
    assert result["fail_count"] == 0
    assert result["pass_count"] == len(C3_2_EXPECTED_ACQUIRES)


def test_batch_marks_controller_not_run_by_default():
    result = batch_static_c3_2()
    assert all(case["controller_status"] == "NOT_RUN" for case in result["cases"])


def test_cli_forwards_camera_to_controller_validate(monkeypatch, tmp_path):
    """--camera must reach controller_validate_case (c3_2 has no agentview)."""
    from tuj.m5_motion.examples import scripted_grasp_validate as cli

    captured: dict = {}

    def fake_controller(object_id, ee, output, **kwargs):
        captured["object_id"] = object_id
        captured["ee"] = ee
        captured["output"] = output
        captured.update(kwargs)
        return {
            "object_id": object_id,
            "static_status": "PASS",
            "controller_status": "PASS",
            "camera": kwargs.get("camera"),
        }

    monkeypatch.setattr(cli, "controller_validate_case", fake_controller)
    code = cli.main(
        [
            "c3_2",
            "--object", "plate_b",
            "--ee", "vac",
            "--controller",
            "--output", str(tmp_path / "out"),
            "--video", str(tmp_path / "out.mp4"),
            "--camera", "robot0_robotview",
        ]
    )
    assert code == 0
    assert captured["object_id"] == "plate_b"
    assert captured["camera"] == "robot0_robotview"
    assert captured["video"] == tmp_path / "out.mp4"


def test_cli_controller_camera_defaults_to_agentview(monkeypatch, tmp_path):
    from tuj.m5_motion.examples import scripted_grasp_validate as cli

    captured: dict = {}

    def fake_controller(*_args, **kwargs):
        captured.update(kwargs)
        return {"static_status": "PASS", "controller_status": "PASS"}

    monkeypatch.setattr(cli, "controller_validate_case", fake_controller)
    assert cli.main(
        [
            "c3_2",
            "--object", "plate_b",
            "--ee", "vac",
            "--controller",
            "--output", str(tmp_path / "out"),
        ]
    ) == 0
    assert captured["camera"] == cli.DEFAULT_CAMERA == "agentview"
