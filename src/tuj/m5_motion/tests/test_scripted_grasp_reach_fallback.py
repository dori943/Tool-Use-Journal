"""Generic catalog PRE/LIFT reach-fallback schedule and behavior."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tuj.m5_motion.grasp_geometry import (
    catalog_lift_reach_standoff_candidates,
    catalog_pre_grasp_reach_minimum_m,
    catalog_pre_grasp_reach_standoff_candidates,
    catalog_reach_fallback_step_m,
    reach_standoff_schedule_m,
)
from tuj.m5_motion.scripted_grasps.frames import transform
from tuj.m5_motion.scripted_grasps.runtime import GraspFailure
from tuj.m5_motion.scripted_grasps.spoon_runtime import (
    approach_spoon,
    lift_with_reach_fallback,
)


def test_reach_schedule_is_relative_step_down_not_absolute_list():
    step = catalog_reach_fallback_step_m()
    floor = catalog_pre_grasp_reach_minimum_m()
    assert step == pytest.approx(0.02)
    assert floor == pytest.approx(0.06)
    assert catalog_pre_grasp_reach_standoff_candidates(0.12) == (
        0.12,
        0.10,
        0.08,
        0.06,
    )
    # Different recipe distance must shift the whole ladder.
    assert catalog_pre_grasp_reach_standoff_candidates(0.15) == (
        0.15,
        0.13,
        0.11,
        0.09,
        0.07,
    )
    assert catalog_lift_reach_standoff_candidates(0.18, minimum_lift_m=0.10) == (
        0.18,
        0.16,
        0.14,
        0.12,
        0.10,
    )
    assert catalog_lift_reach_standoff_candidates(0.14, minimum_lift_m=0.10) == (
        0.14,
        0.12,
        0.10,
    )


def test_reach_schedule_always_includes_original_even_below_floor():
    assert reach_standoff_schedule_m(0.05, step_m=0.02, minimum_m=0.06) == (0.05,)


def test_fruit_a_probe_expected_fallback_distances():
    """Probe: PRE<=0.10 and LIFT<=0.10 work for default fruit recipe distances."""
    pre = catalog_pre_grasp_reach_standoff_candidates(0.12)
    lift = catalog_lift_reach_standoff_candidates(0.18, minimum_lift_m=0.10)
    assert pre[0] == pytest.approx(0.12)
    assert 0.10 in pre
    assert lift[0] == pytest.approx(0.18)
    assert lift[-1] == pytest.approx(0.10)


def _targets(approach_m=0.12, lift_m=0.18):
    grasp = transform([0.4, 0.3, 0.95], rotation=np.diag([1.0, -1.0, -1.0]))
    pre = grasp.copy()
    pre[:3, 3] = grasp[:3, 3] - grasp[:3, 2] * approach_m
    lift = grasp.copy()
    lift[2, 3] += lift_m
    return {"PRE_GRASP": pre, "GRASP": grasp.copy(), "LIFT": lift}


class _FakeContext:
    def __init__(self, move_fn, *, minimum_lift_m=0.10, seed_q=None):
        self._move_fn = move_fn
        self.recipe = type("R", (), {"minimum_lift_m": minimum_lift_m})()
        self.moves = []
        self.data = type("D", (), {"qpos": np.zeros(20)})()
        self.arm_ids = list(range(6))
        self.kinematics = type(
            "K",
            (),
            {
                "solve_all_ik": staticmethod(
                    lambda *a, **k: type(
                        "S",
                        (),
                        {
                            "solutions": (
                                ()
                                if seed_q is None
                                else (type("Sol", (), {"qpos": seed_q})(),)
                            )
                        },
                    )()
                )
            },
        )()

    def move(self, target, stage, opening, cartesian=False, seed_qpos=None):
        self.moves.append(
            {
                "target": target.copy(),
                "stage": stage,
                "opening": opening,
                "cartesian": cartesian,
                "seed_qpos": None if seed_qpos is None else np.asarray(seed_qpos).copy(),
            }
        )
        return self._move_fn(self, target, stage, opening, cartesian, seed_qpos)


def test_original_pre_success_skips_fallback_and_keeps_grasp():
    targets = _targets()
    grasp0 = targets["GRASP"].copy()
    pre0 = targets["PRE_GRASP"].copy()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        return np.zeros(6)

    ctx = _FakeContext(move_fn, seed_q=np.arange(6, dtype=float) * 0.01)
    approach_spoon(ctx, targets, opening=1.0)
    assert len(ctx.moves) == 1
    assert ctx.moves[0]["stage"] == "PRE_GRASP"
    assert np.allclose(targets["PRE_GRASP"], pre0)
    assert np.allclose(targets["GRASP"], grasp0)


def test_grasp_reachable_original_pre_ik_fails_then_shorter_pre_succeeds():
    targets = _targets(approach_m=0.12)
    grasp0 = targets["GRASP"].copy()
    preferred = float(
        np.linalg.norm(targets["PRE_GRASP"][:3, 3] - targets["GRASP"][:3, 3])
    )

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        standoff = float(np.linalg.norm(target[:3, 3] - targets["GRASP"][:3, 3]))
        if standoff >= preferred - 1e-9:
            raise GraspFailure("IK_FAILED: PRE_GRASP no solution")
        return np.ones(6)

    ctx = _FakeContext(move_fn, seed_q=np.arange(6, dtype=float))
    approach_spoon(ctx, targets, opening=0.5)
    used = float(np.linalg.norm(targets["PRE_GRASP"][:3, 3] - targets["GRASP"][:3, 3]))
    assert used == pytest.approx(0.10)
    assert np.allclose(targets["GRASP"], grasp0)
    assert ctx.moves[0]["seed_qpos"] is not None
    assert all(m["stage"] == "PRE_GRASP" for m in ctx.moves)


def test_original_lift_ik_fails_then_shorter_lift_succeeds():
    targets = _targets(lift_m=0.18)
    grasp0 = targets["GRASP"].copy()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        height = float(target[2, 3] - targets["GRASP"][2, 3])
        if height > 0.10 + 1e-9:
            raise GraspFailure("CARTESIAN_IK_FAILED: LIFT")
        return np.full(6, height)

    ctx = _FakeContext(move_fn, minimum_lift_m=0.10)
    q = lift_with_reach_fallback(ctx, targets, opening=-1.0, cartesian=True)
    assert float(targets["LIFT"][2, 3] - targets["GRASP"][2, 3]) == pytest.approx(0.10)
    assert np.allclose(targets["GRASP"], grasp0)
    assert np.allclose(q, np.full(6, 0.10))


def test_truly_unreachable_pre_raises_clean_failure():
    targets = _targets()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        raise GraspFailure("IK_FAILED: PRE_GRASP unreachable")

    ctx = _FakeContext(move_fn)
    with pytest.raises(GraspFailure, match="IK_FAILED: PRE_GRASP"):
        approach_spoon(ctx, targets, opening=1.0)
    assert len(ctx.moves) == len(catalog_pre_grasp_reach_standoff_candidates(0.12))


def test_collision_invalid_shorter_pre_is_rejected():
    targets = _targets()
    grasp0 = targets["GRASP"].copy()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        standoff = float(np.linalg.norm(target[:3, 3] - targets["GRASP"][:3, 3]))
        if standoff >= 0.12 - 1e-9:
            raise GraspFailure("IK_FAILED: PRE_GRASP")
        if standoff >= 0.10 - 1e-9:
            raise GraspFailure("COLLISION_FREE_PATH_NOT_FOUND: PRE_GRASP")
        return np.zeros(6)

    ctx = _FakeContext(move_fn)
    approach_spoon(ctx, targets, opening=1.0)
    used = float(np.linalg.norm(targets["PRE_GRASP"][:3, 3] - targets["GRASP"][:3, 3]))
    assert used == pytest.approx(0.08)
    assert np.allclose(targets["GRASP"], grasp0)
    # Collision candidate was attempted but not accepted as the final PRE.
    assert any(
        float(np.linalg.norm(m["target"][:3, 3] - grasp0[:3, 3])) == pytest.approx(0.10)
        for m in ctx.moves
    )


def test_non_ik_failure_on_original_lift_does_not_fallback():
    targets = _targets()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        raise GraspFailure("COLLISION_FREE_PATH_NOT_FOUND: LIFT")

    ctx = _FakeContext(move_fn)
    with pytest.raises(GraspFailure, match="COLLISION_FREE_PATH_NOT_FOUND"):
        lift_with_reach_fallback(ctx, targets, opening=-1.0)
    assert len(ctx.moves) == 1


def test_fruit_b_style_preferred_pre_regression_keeps_original_pose():
    """fruit_b succeeds at preferred 0.12 m; fallback must not rewrite PRE."""
    targets = _targets(approach_m=0.12)
    pre0 = targets["PRE_GRASP"].copy()
    grasp0 = targets["GRASP"].copy()

    def move_fn(ctx, target, stage, opening, cartesian, seed_qpos):
        return np.zeros(6)

    ctx = _FakeContext(move_fn, seed_q=np.zeros(6))
    approach_spoon(ctx, targets, opening=1.0)
    assert len(ctx.moves) == 1
    assert np.allclose(targets["PRE_GRASP"], pre0)
    assert np.allclose(targets["GRASP"], grasp0)


def test_no_instance_or_task_hardcode_in_reach_fallback_sources():
    roots = [
        Path(__file__).resolve().parents[1] / "scripted_grasps" / "spoon_runtime.py",
        Path(__file__).resolve().parents[1] / "grasp_geometry.py",
    ]
    banned = ("fruit_a", "fruit_b", "c3_2", "0.39")
    for path in roots:
        text = path.read_text(encoding="utf-8")
        # Bound the scan to the new catalog schedule helpers / approach helpers.
        if path.name == "grasp_geometry.py":
            start = text.index("def catalog_reach_fallback_step_m")
            end = text.index("def multi_finger_tabletop_pre_grasp_standoff_candidates")
            text = text[start:end]
        else:
            start = text.index("def _ik_failure_message")
            end = text.index("def preshape_spoon")
            text = text[start:end]
        lowered = text.lower()
        for token in banned:
            assert token not in lowered, f"{token} found in {path.name}"
        assert "(0.10, 0.08, 0.06)" not in text
        assert "(0.15, 0.12, 0.10, 0.08, 0.06)" not in text
