from __future__ import annotations

import numpy as np
import pytest

from tuj.m1_scene import (
    MockBackend,
    PropertyMemory,
    build_m1,
    ground_scene,
    serialize,
)
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
            "bbox_kind": "WORLD_AXIS_ALIGNED_OBSERVED_ENVELOPE",
            "source": "DEPTH_INSTANCE_SEGMENTATION",
            "completeness": "OBSERVED_SURFACES_ONLY",
            "coordinate_frame": {
                "frame_id": "camera_aligned_observation",
                "transform_to_world": {
                    "translation_m": [-0.7, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
    }


def _planar_m1(*, center=(900.0, 100.0, 500.0)):
    x, y = np.meshgrid(
        np.linspace(center[0] - 60.0, center[0] + 60.0, 6),
        np.linspace(center[1] - 50.0, center[1] + 50.0, 6),
    )
    z = np.linspace(center[2] - 0.003, center[2] + 0.003, x.size).reshape(
        x.shape
    )
    m1 = build_m1(
        [
            {
                "name": "plate",
                "cls": "plate",
                "points": np.column_stack((x.ravel(), y.ravel(), z.ravel())),
            }
        ]
    )
    m1["geometry_metadata"] = _m1(center=center)["geometry_metadata"]
    return m1


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


def test_grounded_m1_bbox_remains_valid_for_m5_geometry_contract():
    m1 = _m1()
    center = np.asarray(m1["nodes"][0]["center_mm"], dtype=float)
    offsets = np.asarray(
        [
            [x, y, z]
            for x in (-60.0, 60.0)
            for y in (-50.0, 50.0)
            for z in (-10.0, 10.0)
        ]
    )
    m1["nodes"][0]["_points"] = center + offsets

    ground_scene(m1, backend=MockBackend())
    integrated, report = integrate_m1_geometry(
        _world(),
        serialize(m1),
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert m1["nodes"][0]["geometry"]["center"] == [900.0, 100.0, 500.0]
    assert m1["nodes"][0]["geometry"]["aabb_size"] == [120.0, 100.0, 20.0]
    assert report["planning_safe"] is True
    assert report["records"][0]["status"] == "ALIGNED"
    assert "observation_geometry" in integrated.objects["plate"]


def test_planar_m1_surface_bbox_remains_valid_for_m5_geometry_contract(tmp_path):
    center = (900.0, 100.0, 510.3)
    memory_path = tmp_path / "memory.json"
    ground_scene(
        _planar_m1(center=center),
        backend=MockBackend(),
        memory=PropertyMemory(memory_path, task_id="test-task"),
    )

    m1 = _planar_m1(center=center)
    raw_z_span_mm = float(np.ptp(m1["nodes"][0]["_points"][:, 2]))
    assert raw_z_span_mm == pytest.approx(0.006)
    assert m1["nodes"][0]["bbox_mm"] == [120.0, 100.0, 0.0]

    stats = ground_scene(
        m1,
        backend=MockBackend(),
        memory=PropertyMemory(memory_path, task_id="test-task"),
    )
    assert stats["memory_hits"] == 1
    integrated, report = integrate_m1_geometry(
        _world(),
        serialize(m1),
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is True
    assert report["unsafe_required_object_ids"] == []
    assert report["missing_required_observations"] == []
    assert report["records"][0]["status"] == "ALIGNED"
    assert report["records"][0]["observed_dimensions_m"] == pytest.approx(
        [0.12, 0.10, 0.0]
    )
    assert report["records"][0]["maximum_separation_m"] == pytest.approx(0.0003)
    evidence = integrated.objects["plate"]["observation_geometry"]
    corners = np.asarray(evidence["local_bbox_corners_m"], dtype=float)
    assert corners.shape == (8, 3)
    assert np.all(np.isfinite(corners))
    bounds = spatial_record(integrated.objects["plate"])["world_bounds"]
    assert bounds["source"].endswith("+m1_observation_union")
    assert bounds["aabb_min_m"] == pytest.approx([0.14, 0.05, 0.49])
    assert bounds["aabb_max_m"] == pytest.approx([0.26, 0.15, 0.5103])


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    (
        ("schema", "UNKNOWN"),
        ("bbox_kind", "COMPLETE_OBJECT_AABB"),
        ("completeness", "COMPLETE_OBJECT"),
    ),
)
def test_planar_bbox_requires_explicit_surface_observation_contract(
    metadata_key, metadata_value
):
    m1 = _planar_m1()
    m1["geometry_metadata"][metadata_key] = metadata_value

    _, report = integrate_m1_geometry(
        _world(),
        m1,
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is False
    assert report["unsafe_required_object_ids"] == ["plate"]
    assert report["missing_required_observations"] == ["plate"]
    assert report["records"][0]["status"] == "INVALID_NODE"


@pytest.mark.parametrize(
    "dimensions",
    (
        [120.0, 100.0, -1.0],
        [120.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [120.0, 100.0, float("nan")],
        [120.0, 100.0, float("inf")],
        [120.0, 100.0, "not-a-number"],
        [[120.0], [100.0], [0.0]],
    ),
)
def test_invalid_observed_bbox_dimensions_remain_fail_closed(dimensions):
    m1 = _m1()
    m1["nodes"][0]["geometry"]["aabb_size"] = dimensions

    _, report = integrate_m1_geometry(
        _world(),
        m1,
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is False
    assert report["unsafe_required_object_ids"] == ["plate"]
    assert report["missing_required_observations"] == ["plate"]
    assert report["records"][0]["status"] == "INVALID_NODE"


def test_disjoint_planar_required_geometry_still_stops_planning_contract():
    m1 = _planar_m1(center=(1400.0, 100.0, 500.0))
    # 0912: build_m1 은 노드에 geometry 를 넣지 않는다 (ground_scene 이 채운다).
    # 이 테스트는 백엔드 없이 build_m1 결과를 바로 넘기고 있어 노드가
    # INVALID_NODE 로 떨어졌고, 보려던 DISJOINT 판정까지 가지 못했다. 관측
    # 표면만의 평면 bbox 를 그대로 기하로 붙여 의도한 경로를 타게 한다.
    node = m1["nodes"][0]
    node["geometry"] = {
        "center": list(node["center_mm"]),
        "aabb_size": list(node["bbox_mm"]),
    }

    _, report = integrate_m1_geometry(
        _world(),
        m1,
        required_object_ids={"plate"},
        separation_tolerance_m=0.01,
    )

    assert report["planning_safe"] is False
    assert report["unsafe_required_object_ids"] == ["plate"]
    assert report["missing_required_observations"] == []
    assert report["records"][0]["status"] == "DISJOINT"


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
