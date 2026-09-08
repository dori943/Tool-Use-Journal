from __future__ import annotations

import pytest

from tuj.m5_motion.geometry_evidence import (
    carry_observation_geometry,
    integrate_m1_geometry,
)
from tuj.m5_motion.scene_context import spatial_record
from tuj.m5_motion.schema import RobotState, SceneRef, WorldSnapshot
from tuj.m5_motion.tool_use_journal import apply_world_snapshot_state


def _world(*, position=(0.2, 0.1, 0.5)) -> WorldSnapshot:
    return WorldSnapshot(
        scene=SceneRef(signature="scene"),
        robot_state=RobotState(
            robot_id="ur5e",
            joint_names=["j1"],
            joint_positions_rad=[0.0],
        ),
        objects={
            "plate": {
                "pose": {
                    "frame_id": "world",
                    "position_m": list(position),
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "dimensions_m": [0.1, 0.1, 0.02],
                "anchors": {"center": [0.0, 0.0, 0.0]},
            }
        },
        metadata={"robot_base_world_m": [-0.7, 0.0, 0.4]},
    )


def _m1(*, center=(900.0, 100.0, 500.0)):
    return {
        "nodes": [
            {
                "id": "obj_plate_plate",
                "canonical_id": "plate",
                "class": "plate",
                "center_mm": list(center),
                "bbox_mm": [120.0, 100.0, 20.0],
                "geometry": {
                    "center": list(center),
                    "aabb_size": [120.0, 100.0, 20.0],
                },
            }
        ],
        "edges": [],
        "geometry_metadata": {
            "schema": "M1_GEOMETRY_V2",
            "meters_per_unit": 0.001,
            "coordinate_frame": {
                "frame_id": "camera_aligned_observation",
                "transform_to_world": {
                    "translation_m": [-0.7, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }


def test_m1_bbox_is_normalized_and_bound_to_m5_object():
    integrated, report = integrate_m1_geometry(
        _world(),
        _m1(),
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is True
    assert report["records"][0]["status"] == "ALIGNED"
    evidence = integrated.objects["plate"]["observation_geometry"]
    assert len(evidence["local_bbox_corners_m"]) == 8
    bounds = spatial_record(integrated.objects["plate"])["world_bounds"]
    assert bounds["source"].endswith("+m1_observation_union")
    assert bounds["aabb_min_m"] == pytest.approx([0.14, 0.05, 0.49])
    assert bounds["aabb_max_m"] == pytest.approx([0.26, 0.15, 0.51])


def test_disjoint_required_geometry_stops_planning_contract():
    _, report = integrate_m1_geometry(
        _world(),
        _m1(center=(1400.0, 100.0, 500.0)),
        aliases={"obj_plate_plate": "plate"},
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is False
    assert report["unsafe_required_object_ids"] == ["plate"]
    assert report["records"][0]["status"] == "DISJOINT"


def test_local_observation_envelope_follows_later_object_pose():
    integrated, _ = integrate_m1_geometry(
        _world(),
        _m1(),
        aliases={"obj_plate_plate": "plate"},
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )
    moved = carry_observation_geometry(integrated, _world(position=(0.4, 0.1, 0.5)))

    bounds = spatial_record(moved.objects["plate"])["world_bounds"]
    assert bounds["aabb_min_m"] == pytest.approx([0.34, 0.05, 0.49])
    assert bounds["aabb_max_m"] == pytest.approx([0.46, 0.15, 0.51])
    assert moved.metadata["geometry_evidence"]["planning_safe"] is True


def test_arbitrary_rigid_frame_transform_is_applied_to_bbox_corners():
    m1 = _m1(center=(100.0, -900.0, 500.0))
    node = m1["nodes"][0]
    node["geometry"]["aabb_size"] = [100.0, 120.0, 20.0]
    root_half = 2.0 ** -0.5
    transform = m1["geometry_metadata"]["coordinate_frame"]["transform_to_world"]
    transform["orientation_xyzw"] = [0.0, 0.0, root_half, root_half]

    integrated, report = integrate_m1_geometry(
        _world(),
        m1,
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    bounds = spatial_record(integrated.objects["plate"])["world_bounds"]
    assert bounds["aabb_min_m"] == pytest.approx([0.14, 0.05, 0.49])
    assert bounds["aabb_max_m"] == pytest.approx([0.26, 0.15, 0.51])
    assert report["source_frame_id"] == "camera_aligned_observation"


def test_missing_canonical_identity_is_rejected_without_name_guessing():
    m1 = _m1()
    del m1["nodes"][0]["canonical_id"]

    _, report = integrate_m1_geometry(
        _world(),
        m1,
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is False
    assert report["records"][0]["status"] == "INVALID_NODE"


def test_world_snapshot_restores_named_robot_and_free_object_state():
    import mujoco
    from types import SimpleNamespace

    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody>
          <body name="arm"><joint name="j1" type="hinge"/><geom type="sphere" size=".01"/></body>
          <body name="plate"><freejoint name="plate_free"/><geom type="box" size=".05 .05 .01"/></body>
        </worldbody></mujoco>"""
    )
    data = mujoco.MjData(model)
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=SimpleNamespace(_data=data),
        )
    )
    world = _world(position=(0.3, -0.2, 0.6))
    world.robot_state.joint_positions_rad = [0.75]
    world.objects["plate"]["free_joint_name"] = "plate_free"

    apply_world_snapshot_state(env, world)

    assert data.qpos[0] == pytest.approx(0.75)
    assert data.qpos[1:4] == pytest.approx([0.3, -0.2, 0.6])
    assert data.qpos[4:8] == pytest.approx([1.0, 0.0, 0.0, 0.0])
