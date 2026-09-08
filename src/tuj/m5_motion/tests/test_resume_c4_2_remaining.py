from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tuj.m5_motion.schema import Pose


def _script():
    repository = Path(__file__).resolve().parents[4]
    return runpy.run_path(str(repository / "scripts" / "resume_c4_2_remaining.py"))


def _plan_pose(position=(0.1, 0.2, 0.3)):
    free_pose = SimpleNamespace(
        object_id="milk",
        pose=Pose(
            frame_id="world",
            position_m=position,
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    context = SimpleNamespace(free_object_poses=[free_pose])
    segment = SimpleNamespace(collision_context_before=context)
    return SimpleNamespace(scene_signature="planned-scene", segments=[segment])


def _world_pose(position=(0.1, 0.2, 0.3)):
    return SimpleNamespace(
        objects={
            "milk": {
                "pose": {
                    "position_m": list(position),
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            }
        }
    )


def test_preplanned_rebase_validates_initial_object_poses():
    validate = _script()["_validate_preplanned_world"]

    result = validate(_plan_pose(), _world_pose())

    assert result["source_scene_signature"] == "planned-scene"
    assert result["validated_free_object_count"] == 1


def test_preplanned_rebase_rejects_stale_object_pose():
    validate = _script()["_validate_preplanned_world"]

    with pytest.raises(ValueError, match="collision scene is stale.*milk"):
        validate(_plan_pose(), _world_pose(position=(0.11, 0.2, 0.3)))


def test_raw_state_model_signature_includes_model_layout_and_names():
    signature = _script()["_model_signature"]
    common = dict(nq=6, nv=6, nu=6, nbody=3, njnt=6, ngeom=4, nsite=2)

    first = signature(SimpleNamespace(**common, names=b"bare-model"))
    second = signature(SimpleNamespace(**common, names=b"3f-model"))

    assert first != second
